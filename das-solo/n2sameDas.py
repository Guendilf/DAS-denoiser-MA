import time
import numpy as np
import statistics
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, random_split
from absl import app
from torch.utils.tensorboard import SummaryWriter
import io
from PIL import Image

from import_files import gerate_spezific_das, log_files, mask_random
from import_files import U_Net
from unet_copy import UNet as unet
from import_files import SyntheticNoiseDAS
from import_files import Direct_translation_SyntheticDAS
from scipy import signal

#tensorboard --logdir="E:\Bibiotheken\Dokumente\02 Uni\1 MA\Code\DAS-denoiser-MA\runs"      #PC
#tensorboard --logdir="E:\Bibiotheken\Dokumente\02 Uni\1 MA\runs"                           #vom Server
#tensorboard --logdir="C:\Users\LaAlexPi\Documents\01_Uni\MA\runs\"                         #Laptop

epochs = 1 #2.000 epochen - 1 Epoche = 3424 samples
batchsize = 32
dasChanelsTrain = 11
dasChanelsVal = 11
dasChanelsTest = 11
lr = 0.0001
batchnorm = False
save_model = False

gauge_length = 19.2 #30 for synthhetic?
snr = (-2,4) #(np.log(0.01), np.log(10)) for synthhetic?
slowness = (1/10000, 1/200) #(0.2*10**-3, 10*10**-3) #angabe in m/s, laut paper 0.2 bis 10 km/s     defaault: # (0.0001, 0.005)
sampling = 50.0

modi = 0 #testing diffrent setups


"""
TODO:
- swuish activation funktion with ??? parameter     -> Sigmoid Linear Unit (SiLU) function
- gauge = 19,2
- 50 Hz frequenz
"""
def visualise_spectrum(spectrum_noise, spectrum_denoised):
    fig, axes = plt.subplots(nrows=11, figsize=(10, 20))
    # Iteriere durch jede w-Schicht (zweite Dimension, hier 11)
    for i in range(11):
        axes[i].plot(torch.abs(spectrum_noise[0, 0, i, :]).cpu().detach().numpy(), label='Noise Spectrum')
        axes[i].plot(torch.abs(spectrum_denoised[0, 0, i, :]).cpu().detach().numpy(), label='Denoised Spectrum')
        axes[i].set_title(f'Spectrum - Layer {i+1}')
        axes[i].set_xlabel('Frequency')
        axes[i].set_ylabel('Amplitude')
        axes[i].legend()

    plt.tight_layout()
    #plt.show()
def reconstruct(model, device, noise_images):
    denoised = model(noise_images)
    return denoised

#get bottom line for time-axis
def set_bottom_line(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
# delete black outline of plot
def remove_frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)

def generate_wave_plot(das, noisy_waves, denoised_waves, amps, snrs):
    # Lege den Plot mit 3 Spalten und der gewünschten Anzahl an Zeilen fest
    fig, axs = plt.subplots(3, 3, figsize=(15, 10), gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1, 1, 1]})

    # Gleiche Skalen für alle Plots festlegen
    min_wave = das[0].min().item()
    max_wave = das[0].max().item()

    das = das.to('cpu').detach().numpy()
    noisy_waves = noisy_waves.to('cpu').detach().numpy()
    denoised_waves = denoised_waves.to('cpu').detach().numpy()

    # 1. Spalte: Originalsignal plotten
    axs[1, 0].plot(das[0])
    axs[1, 0].set_title('Original Wave')
    axs[1, 0].set_ylim(min_wave, max_wave)  # Gleiche Skala für y-Achse
    remove_frame(axs[1, 0])  # Entferne den Rahmen
    set_bottom_line(axs[1, 0])  # Zeige nur die untere Linie
    axs[1, 0].yaxis.set_visible(False)  # Deaktiviere y-Achsen für das mittige Diagramm
    axs[1, 0].set_xlabel('Time')

    # Leere Felder in der ersten Spalte
    axs[0, 0].set_axis_off()  # Oberes Feld leer
    axs[2, 0].set_axis_off()  # Unteres Feld leer

    # 2. Spalte: Verrauschte Wellen plotten
    for i, noisy_wave in enumerate(noisy_waves):
        axs[i, 1].plot(noisy_wave[0]/amps[i].item())
        axs[i, 1].set_title(f'Noisy Wave (SNR={snrs[i]})')
        axs[i, 1].set_ylim(min_wave, max_wave)
        remove_frame(axs[i, 1])  # Entferne den Rahmen
        if i < 2:  # Nur im untersten Plot eine x-Achse anzeigen
            axs[i, 1].set_xticks([])
        else:# Für den untersten Plot die Linie und die Beschriftung anzeigen
            set_bottom_line(axs[i, 1])
            axs[i, 1].set_xlabel('Time')
        axs[i, 1].set_yticks([])

    # 3. Spalte: Entstaubte Wellen durch das Modell plotten
    for i, denoised_wave in enumerate(denoised_waves):
        axs[i, 2].plot(denoised_wave[0]/amps[i].item())
        axs[i, 2].set_title(f'Denoised Wave (SNR={snrs[i]})')
        axs[i, 2].set_ylim(min_wave, max_wave)
        remove_frame(axs[i, 2])  # Entferne den Rahmen
        if i < 2:  # Nur im untersten Plot eine x-Achse anzeigen
            axs[i, 2].set_xticks([])
        else:# Für den untersten Plot die Linie und die Beschriftung anzeigen
            set_bottom_line(axs[i, 2])
            axs[i, 2].set_xlabel('Time')
        axs[i, 2].set_yticks([])

    # Beschriftung der x-Achse (nur für den unteren Plot)
    axs[2, 1].set_xlabel('Time')  # X-Achse für zweite Spalte (unten)
    axs[2, 2].set_xlabel('Time')  # X-Achse für dritte Spalte (unten)

    # Layout anpassen
    plt.tight_layout()
    #plt.show()
    return fig

def generate_das_plot(clean_das, all_noisy_waves, all_denoised_waves, amps, snr_indices):
    # Übertrage die Daten auf die CPU
    clean_das_cpu = clean_das.to('cpu').detach().numpy()
    noisy_waves_cpu = all_noisy_waves.to('cpu').detach().numpy()
    denoised_waves_cpu = all_denoised_waves.to('cpu').detach().numpy()

    # Berechne das globale Minimum und Maximum der Daten für eine einheitliche Farbgebung
    vmin = -1e-08 #clean_das_cpu.min() = -0.00000013847445 or -1.3847445e-07
    vmax = 1e-08 #clean_das_cpu.max() = 1.0335354e-07
    vmin = np.percentile(clean_das_cpu, 9)
    vmax = np.percentile(clean_das_cpu, 91)

    # Erstelle das Grid (3 Spalten und 8 Zeilen)
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(8, 3, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1], figure=fig)

    # Spalte 1: Clean DAS (mittig, über vier Zeilen)
    ax_clean = fig.add_subplot(gs[2:6, 0])  # Zeilen 2 bis 5 in Spalte 1
    ax_clean.imshow(clean_das_cpu, aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis', interpolation="antialiased", rasterized=True)
    ax_clean.set_title('Clean DAS' )#if i == 2 else "")
    ax_clean.set_ylabel('Channel Index')
    ax_clean.set_xlabel('Time')
        

    # Spalte 2: Noisy DAS (SNR 0.1 und SNR 10 in Zeilen)
    for i, snr_idx in enumerate(snr_indices):
        ax_noisy = fig.add_subplot(gs[i*4:i*4+4, 1])  # Jeweils zwei Zeilen pro Plot
        ax_noisy.imshow(noisy_waves_cpu[snr_idx] / amps[snr_idx].item(), aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis', interpolation="antialiased", rasterized=True)
        ax_noisy.set_title(f'Denoised DAS (SNR={0.1 if snr_idx == 0 else 10})')
        if i == 1:
            ax_noisy.set_xlabel('Time')
        
    # Spalte 3: Denoised DAS (SNR 0.1 und SNR 10 in Zeilen)
    for i, snr_idx in enumerate(snr_indices):
        ax_denoised = fig.add_subplot(gs[i*4:i*4+4, 2])  # Jeweils zwei Zeilen pro Plot
        ax_denoised.imshow(denoised_waves_cpu[snr_idx] / amps[snr_idx].item(), aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis', interpolation="antialiased", rasterized=True)
        ax_denoised.set_title(f'Denoised DAS (SNR={0.1 if snr_idx == 0 else 10})')
        if i == 1:
            ax_denoised.set_xlabel('Time')

    # Lege das Layout fest und zeige den Plot
    plt.tight_layout()
    #plt.show()
    return fig

def generate_das_plot3(clean_das, all_noisy_waves, all_denoised_waves, amps, snr_indices):
    # Übertrage die Daten auf die CPU
    clean_das_cpu = clean_das.to('cpu').detach().numpy()
    noisy_waves_cpu = all_noisy_waves.to('cpu').detach().numpy()
    denoised_waves_cpu = all_denoised_waves.to('cpu').detach().numpy()

    # Berechne das globale Minimum und Maximum der Daten für eine einheitliche Farbgebung
    vmin = -1e-08 #clean_das_cpu.min() = -0.00000013847445 or -1.3847445e-07
    vmax = 1e-08 #clean_das_cpu.max() = 1.0335354e-07
    vmin = np.percentile(clean_das_cpu, 9)
    vmax = np.percentile(clean_das_cpu, 91)

    # Erstelle das Grid (3 Spalten und 8 Zeilen)
    fig, axs = plt.subplots(3, 3, figsize=(15, 10), gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1, 1, 1]})

    # Spalte 1: Clean DAS (mittig, über vier Zeilen)
    axs[1, 0].imshow(clean_das_cpu, aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis', interpolation="antialiased", rasterized=True)
    axs[1, 0].set_title('Clean DAS')
    #remove_frame(axs[1, 0])  # Entferne den Rahmen
    #set_bottom_line(axs[1, 0])  # Zeige nur die untere Linie
    #axs[1, 0].yaxis.set_visible(False)  # Deaktiviere y-Achsen für das mittige Diagramm
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Channel Index')
    # Leere Felder in der ersten Spalte
    axs[0, 0].set_axis_off()  # Oberes Feld leer
    axs[2, 0].set_axis_off()  # Unteres Feld leer
    
    # Spalte 2: Noisy DAS (SNR 0.1 und SNR 10 in Zeilen)
    for i, snr_idx in enumerate(snr_indices):
        axs[i, 1].imshow(noisy_waves_cpu[i] / amps[i].item(), aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis', interpolation="antialiased", rasterized=True)
        axs[i, 1].set_title(f'Denoised DAS (SNR={snr_idx})')
        if i == 2:
            axs[i, 1].set_xlabel('Time')
            axs[i, 1].set_ylabel('Channel Index')

    # Spalte 3: Denoised DAS (SNR 0.1 und SNR 10 in Zeilen)
    for i, snr_idx in enumerate(snr_indices):
        axs[i, 2].imshow(denoised_waves_cpu[i] / amps[i].item(), aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis', interpolation="antialiased", rasterized=True)
        axs[i, 2].set_title(f'Denoised DAS (SNR={snr_idx})')
        if i == 2:# Für den untersten Plot die Linie und die Beschriftung anzeigen
            axs[i, 2].set_xlabel('Time')
            axs[i, 2].set_ylabel('Channel Index')

    # Lege das Layout fest und zeige den Plot
    plt.tight_layout()
    #plt.show()
    return fig

def save_example_wave(eq_strain_rates_test, model, device, writer, epoch):
    wave = eq_strain_rates_test[6][4200:6248]
    clean_das = gerate_spezific_das(wave, nx=300, nt=2048, eq_slowness=1/(gauge_length*sampling), gauge=gauge_length, fs=sampling)
    clean_das = clean_das.to(device).type(torch.float32)

    SNRs = [0.1, 1, 10]
    all_noise_das = []
    all_denoised_das = []
    amps = []
    for SNR in SNRs:
        noise = np.random.randn(*clean_das.shape)  # Zufälliges Rauschen
        noise = torch.from_numpy(noise).to(device).float()
        snr = 10 ** SNR
        amp = 2 * np.sqrt(snr) / torch.abs(clean_das + 1e-10).max()
        noisy_das = clean_das * amp + noise
        amps.append(amp)
        noisy_das = noisy_das.unsqueeze(0).unsqueeze(0).to(device).float()
        denoised_waves = reconstruct(model, device, noisy_das/noisy_das.std())
        all_noise_das.append(noisy_das.squeeze(0).squeeze(0))
        all_denoised_das.append((denoised_waves*noisy_das.std()).squeeze(0).squeeze(0))
    all_noise_das = torch.stack(all_noise_das)
    all_denoised_das = torch.stack(all_denoised_das)
    amps = torch.stack(amps)
    wave_plot_fig = generate_wave_plot(clean_das, all_noise_das, all_denoised_das, amps, SNRs)
    #das_plot_fig = generate_das_plot(clean_das, all_noise_das, all_denoised_das, amps, snr_indices=[0,2])
    das_plot_fig = generate_das_plot3(clean_das, all_noise_das, all_denoised_das, amps, snr_indices=SNRs)

    buf = io.BytesIO()
    wave_plot_fig.savefig(buf, format='png')
    #plt.show()
    plt.close(wave_plot_fig)
    buf.seek(0)
    img = np.array(Image.open(buf))
    writer.add_image("Image Plot Wave", img, global_step=epoch, dataformats='HWC')
    buf.close()

    buf = io.BytesIO()
    das_plot_fig.savefig(buf, format='png')
    #plt.show()
    plt.close(das_plot_fig)
    buf.seek(0)
    img = np.array(Image.open(buf))
    writer.add_image("Image Plot DAS", img, global_step=epoch, dataformats='HWC')
    buf.close()

    max_intensity=clean_das.max()-clean_das.min()
    for i, snr in enumerate(SNRs):
        mse = torch.mean((clean_das-all_denoised_das[i])**2)
        psnr = 10 * torch.log10((max_intensity ** 2) / mse)
        #calculatte scaled variance (https://figshare.com/articles/software/A_Self-Supervised_Deep_Learning_Approach_for_Blind_Denoising_and_Waveform_Coherence_Enhancement_in_Distributed_Acoustic_Sensing_data/14152277/1?file=26674421) In[13]
        sv = torch.mean((all_noise_das[i] - all_denoised_das[i])**2, dim=-1) / torch.mean((all_noise_das[i])**2, dim=-1)
        sv = torch.mean(sv)
        #fast fourier für Spektrum:
        spectrum_noise = torch.fft.rfft(all_noise_das[i])
        spectrum_denoised = torch.fft.rfft(all_denoised_das[i])
        spectrum_noise_abs = torch.abs(spectrum_noise)  #max amplitude
        spectrum_denoised_abs = torch.abs(spectrum_denoised)
        log_spectrum_noise = torch.log(spectrum_noise_abs + 1e-10)  # log(0) möglich
        log_spectrum_denoised = torch.log(spectrum_denoised_abs + 1e-10)
        lsd = torch.mean((log_spectrum_noise - log_spectrum_denoised) ** 2)
        #Koherenz der Frequenzen TODO: ist das Sinnvoll?
        cross_spectrum = spectrum_noise * torch.conj(spectrum_denoised)
        power_spectrum_a = spectrum_noise_abs ** 2
        power_spectrum_b = spectrum_denoised_abs ** 2
        coherence = (torch.abs(cross_spectrum) ** 2) / (power_spectrum_a * power_spectrum_b + 1e-10)
        coherence = torch.mean(coherence)


        writer.add_scalar(f'Image PSNR of SNR={snr}', psnr, global_step=epoch)
        writer.add_scalar(f'Image Scaled Variance of SNR={snr}', sv, global_step=epoch)
        writer.add_scalar(f'Image LSD of SNR={snr}', lsd, global_step=epoch)
        writer.add_scalar(f'Image Korrelation of SNR={snr}', coherence, global_step=epoch)
    

def calculate_loss(noise_image, model, batch_idx, device, lambda_inv=2):
    mask, marked_points = mask_random(noise_image, maskamount=0.005, mask_size=(1,1))
    #new
    #_,_,mask = Mask.crop_augment_stratified_mask(noise_images, (noise_images.shape[-2],noise_images.shape[-1]), 0.5, augment=False)
    marked_points = torch.sum(mask)

    mask = mask.to(device)
    masked_input = (1-mask) * noise_image + (torch.normal(0, 0.2, size=noise_image.shape).to(device) * mask)
    
    denoised = model(noise_image)
    denoised_mask = model(masked_input)
    mse = torch.nn.MSELoss()
    loss_rec = torch.mean((denoised-noise_image)**2) # mse(denoised, noise_images)
    loss_inv = torch.sum(mask*(denoised-denoised_mask)**2)# mse(denoised, denoised_mask)
    loss = loss_rec + lambda_inv * 1 * (loss_inv/marked_points).sqrt()
    return loss, denoised, denoised_mask #J = count of maked_points

def train(model, device, dataLoader, optimizer, mode, writer, epoch, store_path):
    global modi
    loss_log = []
    psnr_log = []
    scaledVariance_log = []
    lsd_log = []
    coherence_log = []
    save_on_last_epoch=True
    for batch_idx, (noise_images, clean, noise, std, amp, _, _, _) in enumerate(dataLoader):
        clean = clean.to(device).type(torch.float32)
        noise_images = noise_images.to(device).type(torch.float32)
        std = std.to(device).type(torch.float32)
        amp = amp.to(device).type(torch.float32)
        if mode == "train":
            model.train()
            loss, denoised, mask_orig = calculate_loss(noise_images, model, batch_idx, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model.eval()
                loss, denoised, mask_orig = calculate_loss(noise_images, model, batch_idx, device)
                #J-Invariant reconstruction
                denoised = reconstruct(model, device, noise_images)
        #norming
        noise_images *= std
        denoised *= std

        #calculate psnr
        max_intensity=clean.max()-clean.min()
        mse = torch.mean((clean-denoised)**2)
        psnr = 10 * torch.log10((max_intensity ** 2) / mse)
        #calculatte scaled variance (https://figshare.com/articles/software/A_Self-Supervised_Deep_Learning_Approach_for_Blind_Denoising_and_Waveform_Coherence_Enhancement_in_Distributed_Acoustic_Sensing_data/14152277/1?file=26674421) In[13]
        sv = torch.mean((noise_images - denoised)**2, dim=-1) / torch.mean((noise_images)**2, dim=-1)
        sv = torch.mean(sv)#, dim=-1)
        #calculate Log-Spectral Distance (LSD)
        #fast fourier für Spektrum:
        spectrum_noise = torch.fft.rfft(noise_images)
        spectrum_denoised = torch.fft.rfft(denoised)
        spectrum_noise_abs = torch.abs(spectrum_noise)  #max amplitude
        spectrum_denoised_abs = torch.abs(spectrum_denoised)
        log_spectrum_noise = torch.log(spectrum_noise_abs + 1e-10)  # log(0) möglich
        log_spectrum_denoised = torch.log(spectrum_denoised_abs + 1e-10)
        lsd = torch.mean((log_spectrum_noise - log_spectrum_denoised) ** 2)
        #Koherenz der Frequenzen TODO: ist das Sinnvoll?
        cross_spectrum = spectrum_noise * torch.conj(spectrum_denoised)
        power_spectrum_a = spectrum_noise_abs ** 2
        power_spectrum_b = spectrum_denoised_abs ** 2
        coherence = (torch.abs(cross_spectrum) ** 2) / (power_spectrum_a * power_spectrum_b + 1e-10)
        coherence = torch.mean(coherence)

        #log data
        psnr_log.append(round(psnr.item(),3))
        loss_log.append(loss.item())
        scaledVariance_log.append(round(sv.item(),3))
        lsd_log.append(round(lsd.item(),3))
        coherence_log.append(round(coherence.item(),3))
        #visualise_spectrum(spectrum_noise, spectrum_denoised)
        writer.add_scalar(f'Sigma {mode}', noise.std(), global_step=epoch * len(dataLoader) + batch_idx)
    return loss_log, psnr_log, scaledVariance_log, lsd_log, coherence_log

def main(argv=[]):
    print("Starte Programm n2self!")
    if torch.cuda.device_count() == 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:3"
    
    strain_train_dir = "data/DAS/SIS-rotated_train_50Hz.npy"
    strain_test_dir = "data/DAS/SIS-rotated_train_50Hz.npy"

    print("lade Datensätze ...")
    eq_strain_rates = np.load(strain_train_dir)
    eq_strain_rates = eq_strain_rates / eq_strain_rates.std(axis=0) #nomralize (like Z.212 from the paper)
    split_idx = int(0.8 * len(eq_strain_rates))
    eq_strain_rates_train = torch.tensor(eq_strain_rates[:split_idx])
    eq_strain_rates_val = torch.tensor(eq_strain_rates[split_idx:])
    eq_strain_rates_test = np.load(strain_test_dir)
    eq_strain_rates_test = eq_strain_rates_test / eq_strain_rates_test.std(axis=0)
    eq_strain_rates_test = torch.tensor(eq_strain_rates_test)
    dataset = SyntheticNoiseDAS(eq_strain_rates_train, nx=dasChanelsTrain, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=300*batchsize)
    dataset_validate = SyntheticNoiseDAS(eq_strain_rates_val, nx=dasChanelsVal, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=30*batchsize)
    dataset_test = SyntheticNoiseDAS(eq_strain_rates_test, nx=dasChanelsTest, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=30*batchsize)
    
    #dataset = Direct_translation_SyntheticDAS(eq_strain_rates_train, nx=dasChanelsTrain, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=10016, mode="train")
    #dataset_validate = Direct_translation_SyntheticDAS(eq_strain_rates_val, nx=dasChanelsVal, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=992, mode="val")
    #dataset_test = Direct_translation_SyntheticDAS(eq_strain_rates_test, nx=dasChanelsTest, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=992, mode="test")
    
    if len(argv) == 1:
        store_path_root = argv[0]
    else:
        store_path_root = log_files()
    global modi
    #svae best values in [train, val, test] structure
    best_psnr = [-1,-1,-1] #higher = better
    best_sv = [-1,-1,-1] #lower = better
    best_lsd = [-1,-1,-1] #lower = better (Ruaschsignal ähnlich zur rekonstruction im Frequenzbereich)
    best_coherence = [-1,-1,-1] #1 = perfekte Korrelation zwischen Rauschsignale und Rekonstruktion; 0 = keine Korrelation

    for i in range(1):

        store_path = Path(os.path.join(store_path_root, f"n2self-{modi}"))
        store_path.mkdir(parents=True, exist_ok=True)
        tmp = Path(os.path.join(store_path, "tensorboard"))
        tmp.mkdir(parents=True, exist_ok=True)
        tmp = Path(os.path.join(store_path, "models"))
        tmp.mkdir(parents=True, exist_ok=True)

        print("n2self")
        dataLoader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
        dataLoader_validate = DataLoader(dataset_validate, batch_size=batchsize, shuffle=False)
        dataLoader_test = DataLoader(dataset_test, batch_size=batchsize, shuffle=False)

        if modi >= 0:
            model = U_Net(1, first_out_chanel=4, scaling_kernel_size=(1,4), conv_kernel=(3,5), batchNorm=batchnorm, n2self_architecture=False).to(device)
            #model = unet(n_channels=1, feature=4, bilinear=True).to(device)
        #else:
            #model = U_Net(1, first_out_chanel=4, scaling_kernel_size=(1,4), conv_kernel=(3,5), batchNorm=batchnorm, n2self_architecture=True).to(device)
            #model = unet(n_channels=1, feature=4, bilinear=False).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        """
        if method_params['sheduler']:
            lr_lambda = get_lr_lambda(method_params['lr'], method_params['changeLR_steps'], method_params['changeLR_rate'])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        """
        writer = SummaryWriter(log_dir=os.path.join(store_path, "tensorboard"))

        for epoch in tqdm(range(epochs)):

            loss, psnr, scaledVariance_log, lsd_log, coherence_log = train(model, device, dataLoader, optimizer, mode="train", writer=writer, epoch=epoch, store_path=store_path)

            writer.add_scalar('Loss Train', statistics.mean(loss), epoch)
            writer.add_scalar('PSNR Train', statistics.mean(psnr), epoch)
            writer.add_scalar('Scaled Variance Train', statistics.mean(scaledVariance_log), epoch)
            writer.add_scalar('LSD Train', statistics.mean(lsd_log), epoch)
            writer.add_scalar('Korelation Train', statistics.mean(coherence_log), epoch)

            loss_val, psnr_val, scaledVariance_log_val, lsd_log_val, coherence_log_val = train(model, device, dataLoader_validate, optimizer, mode="val", writer=writer, epoch=epoch, store_path=store_path) 

            writer.add_scalar('Loss Val', statistics.mean(loss_val), epoch)
            writer.add_scalar('PSNR Val', statistics.mean(psnr_val), epoch)
            writer.add_scalar('Scaled Variance Val', statistics.mean(scaledVariance_log_val), epoch)
            writer.add_scalar('LSD Val', statistics.mean(lsd_log_val), epoch)
            writer.add_scalar('Korelation Val', statistics.mean(coherence_log_val), epoch)

            if epoch % 5 == 0  or epoch==epochs-1:
                """
                model_save_path = os.path.join(store_path, "models", f"{epoch}-model.pth")
                if save_model  or epoch==epochs-1:
                    torch.save(model.state_dict(), model_save_path)
                else:
                    f = open(model_save_path, "x")
                    f.close()
                """
                save_example_wave(eq_strain_rates_test, model, device, writer, epoch)

            if statistics.mean(psnr) > best_psnr[0]:
                best_psnr[0] = statistics.mean(psnr)
            if statistics.mean(psnr_val) > best_psnr[1]:
                best_psnr[1] = statistics.mean(psnr_val)
            if statistics.mean(scaledVariance_log) < best_sv[0]:
                best_sv[0] = statistics.mean(scaledVariance_log)
            if statistics.mean(scaledVariance_log_val) < best_sv[1]:
                best_sv[1] = statistics.mean(scaledVariance_log_val)
            #TODO: is lower really better? or is higher better in my case
            if statistics.mean(lsd_log) < best_lsd[0]:
                best_lsd[0] = statistics.mean(lsd_log)
            if statistics.mean(lsd_log_val) < best_lsd[1]:
                best_lsd[1] = statistics.mean(lsd_log_val)
            if statistics.mean(coherence_log) > best_coherence[0]:
                best_coherence[0] = statistics.mean(coherence_log)
            if statistics.mean(coherence_log_val) > best_coherence[1]:
                best_coherence[1] = statistics.mean(coherence_log_val)
                
        loss_test, psnr_test, scaledVariance_log_test, lsd_log_test, coherence_log_test = train(model, device, dataLoader_test, optimizer, mode="test", writer=writer, epoch=0, store_path=store_path)
        writer.add_scalar('Loss Test', statistics.mean(loss_test), 0)
        writer.add_scalar('PSNR Test', statistics.mean(psnr_test), 0)
        writer.add_scalar('LSD Test', statistics.mean(lsd_log_test), 0)
        writer.add_scalar('Korelation Test', statistics.mean(coherence_log_test), 0)
        model_save_path = os.path.join(store_path, "models", "last-model.pth")
        torch.save(model.state_dict(), model_save_path)
        best_psnr[2] = statistics.mean(psnr_test)
        best_sv[2] = statistics.mean(scaledVariance_log_test)
        best_lsd[2] = statistics.mean(lsd_log_test)
        best_coherence[2] = statistics.mean(coherence_log_test)
        modi += 1
    print("n2self fertig")
    return

if __name__ == '__main__':
    main()