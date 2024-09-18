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

from import_files import gerate_spezific_das, log_files, bandpass
from import_files import U_Net
from unet_copy import UNet as unet
from import_files import SyntheticNoiseDAS
from import_files import Direct_translation_SyntheticDAS
from scipy import signal

#tensorboard --logdir="E:\Bibiotheken\Dokumente\02 Uni\1 MA\Code\DAS-denoiser-MA\runs"      #PC
#tensorboard --logdir="E:\Bibiotheken\Dokumente\02 Uni\1 MA\runs"                           #vom Server
#tensorboard --logdir="C:\Users\LaAlexPi\Documents\01_Uni\MA\runs\"                         #Laptop

epochs = 500 #2.000 epochen - 1 Epoche = 3424 samples
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

modi = 0 #testing diffrent setups


"""
TODO:
- swuish activation funktion with ??? parameter     -> Sigmoid Linear Unit (SiLU) function
- gauge = 19,2
- 50 Hz frequenz
"""
def reconstruct(model, device, noise_images):
    buffer = torch.zeros_like(noise_images).to(device)
    training_size = 11 #how much chanels were used for training
    num_chunks = noise_images.shape[2] // training_size
    left_chunks = noise_images.shape[2] % training_size
    for chunk_idx in range(num_chunks):
        # Extract the chunk from the larger picture
        start_idx = chunk_idx * training_size
        chunk = noise_images[:, :, start_idx:start_idx+training_size, :]
        for i in range(training_size):
            mask = torch.zeros_like(chunk).to(device)
            mask[:, :, i, :] = 1  # Mask out the i-th channel
            input_image = chunk * (1 - mask)
            j_denoised = model(input_image)
            buffer[:, :, start_idx:start_idx+training_size, :] += j_denoised * mask
    # calculate the left overs when chanels are not a multiplicative of training_size
    for i in range(left_chunks):
        chunk = noise_images[:, :, noise_images.shape[2]-11:, :]
        mask = torch.zeros_like(chunk).to(device)
        mask[:, :, training_size-i-1, :] = 1
        input_image = chunk * (1 - mask)
        j_denoised = model(input_image)
        buffer[:, :, noise_images.shape[2]-11:, :] += j_denoised * mask
    return buffer
"""
def show_das(original, norm=True):
    if isinstance(original, torch.Tensor):
        original = original.to('cpu').detach().numpy()
    original = original[0]
    plt.figure(figsize=(7, 5))
    for i in range(original.shape[1]):
        if norm:
            std = original[0][i].std()
            if std == 0:
                std = 0.000000001
        sr = original[0][i]
        plt.plot(sr + 3*i, c="k", lw=0.5, alpha=1)
        #if every chanle by it self
        #plt.subplot(original.shape[1], 1, i + 1)
        #plt.plot(original[0, i].numpy(), c="k", lw=0.5, alpha=1)
        #plt.title(f'Channel {i + 1}')
        plt.tight_layout()
    plt.show()

def save_das_graph(clean, noise_image, denoised):
    if isinstance(clean, torch.Tensor):
        clean = clean.to('cpu').detach().numpy()
    if isinstance(noise_image, torch.Tensor):
        noise_image = noise_image.to('cpu').detach().numpy()
    if isinstance(denoised, torch.Tensor):
        denoised = denoised.to('cpu').detach().numpy()
    # Create a figure with 2 rows and 4 columns
    fig, axes = plt.subplots(clean.shape[0], 4, figsize=(20, 5*clean.shape[0]))

    # Plot the waves
    for i in range(clean.shape[0]):
        y_min = clean[i].min()
        y_max = clean[i].max()
        y_abs = max(abs(y_min), abs(y_max))
        # Clean wave
        axes[i, 0].plot(clean[i, 0, 0, :], label='Clean')
        axes[i, 0].set_title(f'Clean Wave {i+1}')
        axes[i, 0].set_ylim(-y_abs, y_abs)
        
        # Denoised wave
        axes[i, 1].plot(denoised[i, 0, 0, :], label='Reconstructed')
        axes[i, 1].set_title(f'Reconstructed Wave {i+1}')
        axes[i, 1].set_ylim(-y_abs, y_abs)
        
        # Noise wave
        axes[i, 2].plot(noise_image[i, 0, 0, :], label='Input')
        axes[i, 2].set_title(f'Input Wave {i+1}')
        axes[i, 2].set_ylim(-y_abs, y_abs)
        
        # Overlapping clean and denoised waves
        axes[i, 3].plot(clean[i, 0, 0, :], label='Clean', color='black')
        axes[i, 3].plot(denoised[i, 0, 0, :], label='Reconstructed', color='red')
        axes[i, 3].set_ylim(-y_abs, y_abs)
        axes[i, 3].set_title(f'Clean and Denoised comparison {i+1}')
        axes[i, 3].legend()

    # Ensure the scales in the subplots are the same
    for ax in axes.flat:
        ax.label_outer()
    plt.tight_layout()
    return fig

def save_das_imshow(images, titles):
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    for i, (image, title) in enumerate(zip(images, titles)):
        image = image.to('cpu').detach().numpy()
        axs[i].imshow(image, origin='lower', aspect='auto', cmap='seismic', vmin=-1, vmax=1) #cmap='viridis'
        axs[i].set_title(title)
        axs[i].axis('off')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Bild als Tensor in TensorBoard speichern
    image_tensor = torch.Tensor(np.array(plt.imread(buf)))
    image_tensor = image_tensor.permute(2, 0, 1)  # Channels First (C, H, W)
    return image_tensor, fig
 
def saveAndPicture(psnr, clean, noise_images, denoised, mask, std, mode, writer, epoch, len_dataloader, batch_idx, model, store, best):
    #imshow
    noise_images_mask = noise_images * mask
    denoised_mask = denoised * mask
    images = [clean[0, 0, :, :], denoised[0, 0, :, :], noise_images[0, 0, :, :], noise_images_mask[0, 0, :, :], denoised_mask[0, 0, :, :]]
    titles = ['Clean', 'Denoised', 'Input', 'Input * Mask', 'Denoised * Mask']
    image_imshow, imshow_fig = save_das_imshow(images, titles)
    #plt.show()
    plt.close(imshow_fig)

    #graphen
    clean_tmp = clean[:2]
    clean_tmp = clean_tmp[:,:,:,0:512]
    noise_images = noise_images[:2]
    noise_images = noise_images[:,:,:,0:512]
    denoised = denoised[:2]
    denoised = denoised[:,:,:,0:512]
    chanels = []
    for i in range(clean_tmp.shape[0]):
        for j in range(clean_tmp.shape[2]):
            if mask[i,0,j,0] == 1:
                chanels.append(j)
                break
    clean_tmp = clean_tmp[:,:,chanels,:]
    noise_images = noise_images[:,:,chanels,:]
    denoised = denoised[:,:,chanels,:]
    fig = save_das_graph(clean_tmp, noise_images, denoised)
    # Speichere das Bild in TensorBoard
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    #plt.show()
    plt.close(fig)
    buf.seek(0)
    image_graph = np.array(Image.open(buf))
        

    if mode == "train":
        writer.add_image('Graph Denoised Training', image_graph, global_step=epoch * len_dataloader + batch_idx, dataformats='HWC')
        writer.add_image('Imshow Denoised Training', image_imshow, global_step=epoch * len_dataloader + batch_idx)
    elif mode == "val":
        writer.add_image('Graph Denoised Validation', image_graph, global_step=epoch * len_dataloader + batch_idx, dataformats='HWC')
        writer.add_image('Imshow Denoised Validation', image_imshow, global_step=epoch * len_dataloader + batch_idx)
    else:
        writer.add_image('Graph Denoised Test', image_graph, global_step=epoch * len_dataloader + batch_idx, dataformats='HWC')
        writer.add_image('Imshow Denoised Test', image_imshow, global_step=epoch * len_dataloader + batch_idx)
    #TODO:
    #imshow(denoised) und co aspecratio, vmin, vmax
    
    if not best:
        return
    if "test" not in mode:
        print(f"best model found with psnr: {psnr}")
        model_save_path = os.path.join(store, "models", f"{round(psnr, 1)}-{mode}-{epoch}-{batch_idx}.pth")
        if save_model:
            torch.save(model.state_dict(), model_save_path)
        else:
            f = open(model_save_path, "x")
            f.close()
    
"""
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

def save_example_wave(eq_strain_rates_test, model, device, writer, epoch):
    wave = eq_strain_rates_test[6][4200:6248]
    clean_das = gerate_spezific_das(wave, nx=300, nt=2048, eq_slowness=1/(19.2*50.0), gauge=19.2, fs=50.0)
    clean_das = clean_das.to(device).type(torch.float32)

    SNRs = [0.1, 1, 10]
    all_noise_das = []
    all_denoised_das = []
    amps = []
    for SNR in SNRs:
        noise = np.random.randn(len(wave))  # Zufälliges Rauschen
        noise = torch.from_numpy(noise)
        snr = 10 ** SNR
        amp = 2 * np.sqrt(snr) / torch.abs(wave + 1e-10).max()
        noisy_wave = wave * amp + noise
        amps.append(amp)
        noisy_wave = gerate_spezific_das(noisy_wave,  nx=300, nt=2048, eq_slowness=1/(19.2*50.0),
                    gauge=19.2, fs=50.0, station=None, start=None)
        noisy_wave = noisy_wave.unsqueeze(0).unsqueeze(0).to(device)
        denoised_waves = reconstruct(model, device, noisy_wave)
        all_noise_das.append(noisy_wave.squeeze(0).squeeze(0))
        all_denoised_das.append(denoised_waves.squeeze(0).squeeze(0))
    all_noise_das = torch.stack(all_noise_das)
    all_denoised_das = torch.stack(all_denoised_das)
    amps = torch.stack(amps)
    wave_plot_fig = generate_wave_plot(clean_das, all_noise_das, all_denoised_das, amps, SNRs)
    das_plot_fig = generate_das_plot(clean_das, all_noise_das, all_denoised_das, amps, snr_indices=[0,2])



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
        writer.add_scalar(f'Image PSNR of SNR={snr}', psnr, global_step=epoch)
        writer.add_scalar(f'Image Scaled Variance of SNR={snr}', sv, global_step=epoch)
    

def calculate_loss(noise_image, model, batch_idx):
    #masked_noise_image, mask = Mask.n2self_mask(noise_image, batch_idx)
    mask = torch.zeros_like(noise_image).to(noise_image.device)
    for i in range(mask.shape[0]):
        mask[i, :, np.random.randint(0, mask.shape[2]), :] = 1
    masked_noise_image = (1-mask) * noise_image
    denoised = model(masked_noise_image)
    return torch.nn.MSELoss()(denoised*(mask), noise_image*(mask)), denoised, mask

def train(model, device, dataLoader, optimizer, mode, writer, epoch, store_path, bestPsnr):
    global modi
    loss_log = []
    psnr_log = []
    scaledVariance_log = []
    save_on_last_epoch=True
    for batch_idx, (noise_images, clean, noise, std, amp, _, _, _) in enumerate(dataLoader):
        clean = clean.to(device).type(torch.float32)
        noise_images = noise_images.to(device).type(torch.float32)
        std = std.to(device).type(torch.float32)
        amp = amp.to(device).type(torch.float32)
        if mode == "train":
            model.train()
            loss, denoised, mask_orig = calculate_loss(noise_images, model, batch_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model.eval()
                loss, denoised, mask_orig = calculate_loss(noise_images, model, batch_idx)
                #J-Invariant reconstruction
                denoised = reconstruct(model, device, noise_images)
        #norming
        if modi == 0:
            noise_images *= std
            denoised *= std
        if modi == 1:
            clean /= amp.view(amp.shape[0],1,1,1)
            noise_images /= amp.view(amp.shape[0],1,1,1)
            denoised /= amp.view(amp.shape[0],1,1,1)
        #calculate psnr
        max_intensity=clean.max()-clean.min()
        mse = torch.mean((clean-denoised)**2)
        psnr = 10 * torch.log10((max_intensity ** 2) / mse)
        #calculatte scaled variance (https://figshare.com/articles/software/A_Self-Supervised_Deep_Learning_Approach_for_Blind_Denoising_and_Waveform_Coherence_Enhancement_in_Distributed_Acoustic_Sensing_data/14152277/1?file=26674421) In[13]
        sv = torch.mean((noise_images - denoised)**2, dim=-1) / torch.mean((noise_images)**2, dim=-1)
        sv = torch.mean(sv)#, dim=-1)

        #log data
        psnr_log.append(round(psnr.item(),3))
        loss_log.append(loss.item())
        scaledVariance_log.append(round(sv.item(),3))
        writer.add_scalar(f'Sigma {mode}', noise.std(), global_step=epoch * len(dataLoader) + batch_idx)
        """
        if batch_idx % 100 == 0 or batch_idx == len(dataLoader)-1:
            saveAndPicture(psnr.item(), clean, noise_images, denoised, mask_orig, std, mode, writer, epoch, len(dataLoader), batch_idx, model, store_path, True)
        elif mode == "test" and batch_idx%50 == 0:
            saveAndPicture(psnr.item(), clean, noise_images, denoised, mask_orig, std, mode, writer, epoch, len(dataLoader), batch_idx, model, store_path, True)
        """
    return loss_log, psnr_log, scaledVariance_log, bestPsnr

def main(arggv):
    print("Starte Programm!")
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
    dataset = SyntheticNoiseDAS(eq_strain_rates_train, nx=dasChanelsTrain, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=793*batchsize)
    dataset_validate = SyntheticNoiseDAS(eq_strain_rates_val, nx=dasChanelsVal, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=30*batchsize)
    dataset_test = SyntheticNoiseDAS(eq_strain_rates_test, nx=dasChanelsTest, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=30*batchsize)
    
    #dataset = Direct_translation_SyntheticDAS(eq_strain_rates_train, nx=dasChanelsTrain, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=10016, mode="train")
    #dataset_validate = Direct_translation_SyntheticDAS(eq_strain_rates_val, nx=dasChanelsVal, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=992, mode="val")
    #dataset_test = Direct_translation_SyntheticDAS(eq_strain_rates_test, nx=dasChanelsTest, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=992, mode="test")
    
    

    store_path_root = log_files()
    global modi
    for i in range(2):

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

        bestPsnrTrain=0
        bestPsnrVal=0
        bestPsnrTest=0
        for epoch in tqdm(range(epochs)):
            save_example_wave(eq_strain_rates_test, model, device, writer, epoch)
            break
            loss, psnr, scaledVariance_log, bestPsnrTrain = train(model, device, dataLoader, optimizer, mode="train", writer=writer, epoch=epoch, store_path=store_path, bestPsnr=bestPsnrTrain)

            writer.add_scalar('Loss Train', statistics.mean(loss), epoch)
            writer.add_scalar('PSNR Train', statistics.mean(psnr), epoch)
            writer.add_scalar('Scaled Variance Train', statistics.mean(scaledVariance_log), epoch)

            loss_val, psnr_val, scaledVariance_log_val, bestPsnrVal = train(model, device, dataLoader_validate, optimizer, mode="val", writer=writer, epoch=epoch, store_path=store_path, bestPsnr=bestPsnrVal) 

            writer.add_scalar('Loss Val', statistics.mean(loss_val), epoch)
            writer.add_scalar('PSNR Val', statistics.mean(psnr_val), epoch)
            writer.add_scalar('Scaled Variance Val', statistics.mean(scaledVariance_log_val), epoch)

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

        loss_test, psnr_test, scaledVariance_log_test, bestPsnrTest = train(model, device, dataLoader_test, optimizer, mode="test", writer=writer, epoch=0, store_path=store_path, bestPsnr=bestPsnrTest)
        writer.add_scalar('Loss Test', statistics.mean(loss_test), 0)
        writer.add_scalar('PSNR Test', statistics.mean(psnr_test), 0)
        writer.add_scalar('Scaled Variance Test', statistics.mean(scaledVariance_log_test), 0)
        model_save_path = os.path.join(store_path, "models", "last-model.pth")
        torch.save(model.state_dict(), model_save_path)
        modi += 1
    print("fertig")

if __name__ == '__main__':
    app.run(main)