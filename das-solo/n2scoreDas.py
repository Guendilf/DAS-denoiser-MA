import time
import numpy as np
import statistics
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, random_split
from absl import app
from torch.utils.tensorboard import SummaryWriter
import io
from PIL import Image
import h5py
from scipy import signal


from import_files import gerate_spezific_das, log_files, bandpass, tv_norm
from import_files import U_Net, Sebastian_N2SUNet
from unet_copy import UNet as unet
from import_files import SyntheticNoiseDAS, RealDAS
from utils_DAS import moving_window_semblance, semblance
from plots import generate_wave_plot, generate_das_plot3, generate_real_das_plot
from utils_DAS import compute_moving_coherence
import pandas as pd

#tensorboard --logdir="E:\Bibiotheken\Dokumente\02 Uni\1 MA\Code\DAS-denoiser-MA\runs"      #PC
#tensorboard --logdir="E:\Bibiotheken\Dokumente\02 Uni\1 MA\runs"                           #vom Server
#tensorboard --logdir="C:\Users\LaAlexPi\Documents\01_Uni\MA\runs\"                         #Laptop

epochs = 100 #2.000 epochen - 1 Epoche = 3424 samples
realEpochs = 100
batchsize = 32
dasChanelsTrain = 100
dasChanelsVal = 100
dasChanelsTest = 100
nt = 2048

lr = 0.0001
batchnorm = False
save_model = False

gauge_length = 10
channel_spacing = 20#oder 19.2
snr = (-2,4) #(np.log(0.01), np.log(10)) for synthhetic?
slowness = (1/10000, 1/200) #(0.2*10**-3, 10*10**-3) #angabe in m/s, laut paper 0.2 bis 10 km/s     defaault: # (0.0001, 0.005)
sampling = 50.0

modi = 0 #testing diffrent setups


"""
TODO:
- swuish activation funktion with ??? parameter     -> Sigmoid Linear Unit (SiLU) function
- channel_spacing = 19,2
- 50 Hz frequenz
"""



def reconstruct(noise_images, model, tweedie, train=False):
    with torch.no_grad():
        model.eval()
        vectorMap =  model(noise_images)
    denoised = []
    best_tv, best_sigma = -1,-1
    if 'real' in tweedie:
        _, alpha, beta = tweedie.split('_')
        alpha = int(alpha)
        beta = int(beta)
        best_tv, best_sigma = evaluateSigma(noise_images, vectorMap)
        denoised.append(noise_images + best_sigma**2 * vectorMap)#gaus
        if train:
            return denoised, best_tv, best_sigma
        denoised.append((noise_images + 1/2) * torch.exp(vectorMap))#poisson
        denoised.append((beta * noise_images)/((1-alpha) - noise_images * vectorMap))#gamma
        denoised.append(torch.exp(vectorMap) / (1 + torch.exp(vectorMap)))#bernoulli
        denoised.append(-vectorMap)#expo
    elif 'gaus' in tweedie:
        best_tv, best_sigma = evaluateSigma(noise_images, vectorMap)
        denoised = noise_images + best_sigma**2 * vectorMap
    elif 'poisson' in tweedie:
        denoised = (noise_images + 1/2) * torch.exp(vectorMap)
    elif 'gaamma' in tweedie:
        _, alpha, beta = tweedie.split('_')
        alpha = int(alpha)
        beta = int(beta)
        denoised = (beta * noise_images)/((1-alpha) - noise_images * vectorMap)
    elif 'bernoulli' in tweedie:
        denoised = torch.exp(vectorMap) / (1 + torch.exp(vectorMap))
    elif 'expo' in tweedie:
        denoised = -vectorMap
    else:
        raise ValueError(f'Tweedie formel "{tweedie}" not defined')
    
    return denoised, best_tv, best_sigma

def save_example_wave(clean_das_original, model, device, writer, epoch, real_denoised=None, vmin=-1, vmax=1, mask_methode='channel_1'):
    SNRs = [0.1, 1, 10]
    all_noise_das = []
    all_denoised_das = []
    amps = []
    stds = []
    if clean_das_original.shape[0] == 300:
        for SNR in SNRs:
            noise = np.random.randn(*clean_das_original.shape)  # Zufälliges Rauschen
            noise = torch.from_numpy(noise).to(device).float()
            snr = 10 ** SNR
            amp = 2 * np.sqrt(snr) / torch.abs(clean_das_original + 1e-10).max()
            amps.append(amp)
            clean_das = clean_das_original * amp
            noisy_das = clean_das + noise
            noisy_das = noisy_das.unsqueeze(0).unsqueeze(0).to(device).float()
            denoised_waves, _, _ = reconstruct(noisy_das/noisy_das.std(), model, tweedie=mask_methode)
            all_noise_das.append(noisy_das.squeeze(0).squeeze(0))
            all_denoised_das.append((denoised_waves*noisy_das.std()).squeeze(0).squeeze(0))
        
        all_noise_das = torch.stack(all_noise_das)
        all_denoised_das = torch.stack(all_denoised_das)
        amps = torch.stack(amps)
        all_semblance = semblance(all_denoised_das.unsqueeze(1).to(device))#fast implementation without moving window correction
        """
        all_semblance = []
        for das in all_denoised_das: #tqdm?
            all_semblance.append(torch.from_numpy(moving_window_semblance(np.swapaxes(das.cpu().numpy(), 0, 1), window=(60//channel_spacing,25))))
        all_semblance = torch.stack(all_semblance)
        """
        #normalise for picture making
        clean_das = clean_das/amps[-1]
        all_noise_das = all_noise_das/amps[:,None,None]
        all_denoised_das = all_denoised_das/amps[:,None,None]

        wave_plot_fig = generate_wave_plot(clean_das, all_noise_das, all_denoised_das, SNRs)
        das_plot_fig = generate_das_plot3(clean_das, all_noise_das, all_denoised_das, all_semblance, snr_indices=SNRs, vmin=vmin, vmax=vmax)

        buf = io.BytesIO()
        wave_plot_fig.savefig(buf, format='png')
        #plt.show()
        plt.close(wave_plot_fig)
        buf.seek(0)
        img = np.array(Image.open(buf))
        writer.add_image(f"Image Plot Wave {mask_methode}", img, global_step=epoch, dataformats='HWC')
        buf.close()

        buf = io.BytesIO()
        das_plot_fig.savefig(buf, format='png')
        #plt.show()
        plt.close(das_plot_fig)
        buf.seek(0)
        img = np.array(Image.open(buf))
        writer.add_image(f"Image Plot DAS {mask_methode}", img, global_step=epoch, dataformats='HWC')
        buf.close()
    else:
        clean_das = clean_das_original.detach().cpu().numpy()
        all_noise_das = clean_das_original.detach().cpu().numpy()
        all_semblance = semblance(torch.from_numpy(real_denoised).to(device)).squeeze(0).cpu().numpy() #fast implementation without moving window correction
        #all_semblance = moving_window_semblance(np.swapaxes(real_denoised, 0, 1))
        SNRs = ['original']
        channel_idx_1 = 920 // 5 #weil in der bild generierung jeder 5. Kanal überspringen wird
        channel_idx_2 = 3000 // 5
        min_wave = min(clean_das[channel_idx_1].min(),clean_das[channel_idx_2].min())
        max_wave = max(clean_das[channel_idx_1].max(),clean_das[channel_idx_2].max())

        cc_clean = compute_moving_coherence(clean_das, 11) #11 weil 11 Kanäle in training?
        cc_rec = compute_moving_coherence(real_denoised, 11) #11 weil 11 Kanäle in training?
        cc_gain_rec = cc_rec / cc_clean
        
        fig = generate_real_das_plot(clean_das, real_denoised, all_semblance, channel_idx_1, channel_idx_2, cc_gain_rec, vmin, vmax, min_wave, max_wave)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = np.array(Image.open(buf))
        writer.add_image(f"Image Plot DAS Real {mask_methode}", img, global_step=epoch, dataformats='HWC')
        buf.close()

        clean_das = torch.from_numpy(clean_das)
        all_noise_das = torch.from_numpy(all_noise_das)
        all_denoised_das = torch.from_numpy(real_denoised)



    max_intensity=clean_das.max()-clean_das.min()
    for i, snr in enumerate(SNRs):
        mse = torch.mean((clean_das-all_denoised_das[i])**2)
        psnr = 10 * torch.log10((max_intensity ** 2) / mse)
        #calculatte scaled variance (https://figshare.com/articles/software/A_Self-Supervised_Deep_Learning_Approach_for_Blind_Denoising_and_Waveform_Coherence_Enhancement_in_Distributed_Acoustic_Sensing_data/14152277/1?file=26674421) In[13]
        sv = torch.mean((clean_das - all_denoised_das[i])**2, dim=-1) / torch.mean((clean_das)**2, dim=-1)
        sv = torch.mean(sv)
        #fast fourier für Spektrum:
        spectrum_noise = torch.fft.rfft(clean_das)
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
    
def evaluateSigma(noise_image, vector):
    sigmas = torch.linspace(0.1, 0.7, 61)
    quality_metric = []
    for sigma in sigmas:
        
        simple_out = noise_image + sigma**2 * vector.detach()
        simple_out = (simple_out + 1) / 2
        quality_metric += [tv_norm(simple_out).item()]
    
    sigmas = sigmas.numpy()
    quality_metric = np.array(quality_metric)
    best_idx = np.argmin(quality_metric)
    return quality_metric[best_idx], sigmas[best_idx]

def calculate_loss(noise_image, model, batch_idx, device, len_dataLoader):
    sigma_min=0.01
    sigma_max=0.3
    q=batch_idx/len_dataLoader

    u = torch.randn_like(noise_image).to(device)
    sigma_a = sigma_max*(1-q) + sigma_min*q
    vectorMap = model(noise_image+sigma_a*u)
    loss = torch.mean((sigma_a * vectorMap + u)**2)#depends on methode (whatt noise to use in tweedie)

    return loss, vectorMap

def train(model, device, dataLoader, optimizer, mode, writer, epoch, tweedie='gaus'):
    global modi
    loss_log = []
    psnr_log = []
    scaledVariance_log = []
    lsd_log = []
    coherence_log = []
    best_sigmas = []    #only for Score in validation + test
    all_tvs = []     #only for Score in validation + test
    true_sigma = []
    ccGain_log = []
    for batch_idx, (noise_images, clean, noise, std, amp, _, _, _) in enumerate(dataLoader):
        clean = clean.to(device).type(torch.float32)
        noise_images = noise_images.to(device).type(torch.float32)
        std = std.to(device).type(torch.float32)
        amp = amp.to(device).type(torch.float32)
        true_sigma.append(noise.std())

        if mode == "train":
            model.train()
            loss, vectorMap = calculate_loss(noise_images, model, batch_idx, device, len(dataLoader))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model.eval()
                loss, _ = calculate_loss(noise_images, model, batch_idx, device, len(dataLoader))
        denoised, best_tv, best_sigma = reconstruct(noise_images, model, tweedie, train=True)
        if 'real' in tweedie:
            denoised = denoised[0]
        best_sigmas.append(best_sigma)
        all_tvs.append(best_tv)

        #norming war eigentlich an der Stelle
        noise_images *= std
        denoised *= std

        #calculate psnr
        max_intensity=clean.max()-clean.min()
        mse = torch.mean((clean-denoised)**2)
        psnr = 10 * torch.log10((max_intensity ** 2) / mse)
        #calculatte scaled variance (https://figshare.com/articles/software/A_Self-Supervised_Deep_Learning_Approach_for_Blind_Denoising_and_Waveform_Coherence_Enhancement_in_Distributed_Acoustic_Sensing_data/14152277/1?file=26674421) In[13]
        sv = torch.mean((clean - denoised)**2, dim=-1) / torch.mean((clean)**2, dim=-1)
        sv = torch.mean(sv)#, dim=-1)
        """
        #calculate Log-Spectral Distance (LSD)
        #fast fourier für Spektrum:
        spectrum_noise = torch.fft.rfft(clean)
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
        """
        #cc-gain
        #if 'val' in mode or 'test' in mode:
            #cc_clean = compute_moving_coherence(clean[0][0].cpu().detach().numpy(), dasChanelsTrain) #11 weil 11 Kanäle in training?
            #cc_rec = compute_moving_coherence(denoised[0][0].cpu().detach().numpy(), dasChanelsTrain) #11 weil 11 Kanäle in training?
            #cc_value = (np.mean(cc_rec / cc_clean))
        #else:
        cc_value = -1
        lsd = torch.tensor(-1.0)
        coherence = torch.tensor(-1.0)
        #log data
        ccGain_log.append(round(cc_value,3))
        psnr_log.append(round(psnr.item(),3))
        loss_log.append(loss.item())
        scaledVariance_log.append(round(sv.item(),3))
        lsd_log.append(round(lsd.item(),3))
        coherence_log.append(round(coherence.item(),3))
        #visualise_spectrum(spectrum_noise, spectrum_denoised)
        if epoch < epochs:
            writer.add_scalar(f'Sigma {mode}', noise.std(), global_step=epoch * len(dataLoader) + batch_idx)

    ture_sigma_line = np.mean(true_sigma)
    if mode == "train":
        writer.add_scalar('Train true sigma', ture_sigma_line, epoch )
        writer.add_scalar('Train sigma', np.mean(best_sigmas), epoch )
        writer.add_scalar('Train tv', np.mean(best_tv), epoch)
    elif mode == "val":
        writer.add_scalar('Validation true sigma', ture_sigma_line, epoch )
        writer.add_scalar('Validation sigma', np.mean(best_sigmas), epoch )
        writer.add_scalar('Validation tv', np.mean(best_tv), epoch)
    else:
        writer.add_scalar('Test ture sigma', ture_sigma_line, epoch )
        writer.add_scalar('Test sigma', np.mean(best_sigmas), epoch )
        writer.add_scalar('Test tv', np.mean(best_tv), epoch)
    return loss_log, psnr_log, scaledVariance_log, lsd_log, coherence_log, ccGain_log

def main(argv=[]):
    print("Starte Programm n2core!")
    if torch.cuda.device_count() == 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:2"
    
    strain_train_dir = "data/DAS/SIS-rotated_train_50Hz.npy"
    strain_test_dir = "data/DAS/SIS-rotated_test_50Hz.npy"

    print("lade Synthetische Datensätze ...")
    #"""
    eq_strain_rates = np.load(strain_train_dir)
    # Normalise waveforms
    N_ch, N_t = eq_strain_rates.shape
    t_slice = slice(N_t//4, 3*N_t//4)
    stds = eq_strain_rates[:, t_slice].std(axis=1, keepdims=True)
    eq_strain_rates = eq_strain_rates / stds

    split_idx = int(0.8 * len(eq_strain_rates))
    eq_strain_rates_train = torch.tensor(eq_strain_rates[:split_idx])
    eq_strain_rates_val = torch.tensor(eq_strain_rates[split_idx:])

    eq_strain_rates_test = np.load(strain_test_dir)
    # Normalise waveforms
    N_ch, N_t = eq_strain_rates_test.shape
    t_slice = slice(N_t//4, 3*N_t//4)
    stds = eq_strain_rates_test[:, t_slice].std(axis=1, keepdims=True)
    eq_strain_rates_test = eq_strain_rates_test / stds



    eq_strain_rates_test = torch.tensor(eq_strain_rates_test)
    dataset = SyntheticNoiseDAS(eq_strain_rates_train, nx=dasChanelsTrain, eq_slowness=slowness, log_SNR=snr, gauge=channel_spacing, size=300*batchsize)
    dataset_validate = SyntheticNoiseDAS(eq_strain_rates_val, nx=dasChanelsVal, eq_slowness=slowness, log_SNR=snr, gauge=channel_spacing, size=30*batchsize)
    dataset_test = SyntheticNoiseDAS(eq_strain_rates_test, nx=dasChanelsTest, eq_slowness=slowness, log_SNR=snr, gauge=channel_spacing, size=30*batchsize)
    #"""
    #---------------real daten laden----------------
    #"""
    print("lade Reale Datensätze ...")
    train_path = "Server_DAS/real_train/"
    test_path = "Server_DAS/real_test/"
    train_path = sorted([train_path + f for f in os.listdir(train_path)])
    test_paths = sorted([test_path + f for f in os.listdir(test_path)])

    train_real_data = []
    for i, p in enumerate(train_path):
        with h5py.File(p, 'r') as hf:
            DAS_sample = hf['DAS'][81:]
            if channel_spacing == 20:
                DAS_sample = DAS_sample[::5]
            train_real_data.append(DAS_sample)
    train_real_data = np.stack(train_real_data)
    gutter = 1000
    train_real_data = np.pad(train_real_data, ((0,0),(0,0),(gutter,gutter)), mode='constant', constant_values=0)
    chunks = np.array_split(train_real_data, 10)
    processed_chunks = [bandpass(chunk, low=1.0, high=10.0, fs=sampling, gutter=gutter) for chunk in chunks]
    train_real_data = np.concatenate(processed_chunks, axis=0)
    batch, N_ch, N_t = train_real_data.shape
    #effiziente for schleife (identisch zu oben mit wv)
    t_slice = slice(N_t//4, 3*N_t//4)
    stds = train_real_data[:, :, t_slice].std(axis=2, keepdims=True)
    train_real_data_all = train_real_data / stds
    train_real_data = train_real_data_all[:20,:,:]
    val_real_data = train_real_data_all[20:,:,:]

    test_real_data = []
    for i, p in enumerate(test_paths):
        with h5py.File(p, 'r') as hf:
            DAS_sample = hf['DAS'][81:]
            if channel_spacing == 20:
                DAS_sample = DAS_sample[::5]
            test_real_data.append(DAS_sample)
    test_real_data = np.stack(test_real_data)
    gutter = 1000
    test_real_data = np.pad(test_real_data, ((0,0),(0,0),(gutter,gutter)), mode='constant', constant_values=0)
    chunks = np.array_split(test_real_data, 5)
    processed_chunks = [bandpass(chunk, low=1.0, high=10.0, fs=sampling, gutter=gutter) for chunk in chunks]
    test_real_data = np.concatenate(processed_chunks, axis=0)
    batch, N_ch, N_t = test_real_data.shape
    #effiziente for schleife (identisch zu oben mit wv)
    t_slice = slice(N_t//4, 3*N_t//4)
    stds = test_real_data[:, :, t_slice].std(axis=2, keepdims=True)
    test_real_data = test_real_data / stds

    real_dataset = RealDAS(train_real_data, nx=dasChanelsTrain, nt=nt, size=200*batchsize)
    real_dataset_val = RealDAS(val_real_data, nx=dasChanelsVal, nt=nt, size=20*batchsize)
    real_dataset_test = RealDAS(test_real_data, nx=dasChanelsTest, nt=nt, size=20*batchsize)
    #"""
    print("Datensätze geladen!")

    wave = eq_strain_rates_test[6][4200:6248]
    picture_DAS_syn = gerate_spezific_das(wave, nx=300, nt=2048, eq_slowness=1/(500), gauge=channel_spacing, fs=sampling)
    picture_DAS_syn = picture_DAS_syn.to(device).type(torch.float32)
    picture_DAS_real1 = torch.from_numpy(test_real_data[2][:,4576:]).to(device).type(torch.float32)

    
  
    if len(argv) == 1:
        store_path_root = argv[0]
    else:
        store_path_root = log_files()
    global modi
   
    masking_methodes=['real_1_1']#, 'gaus']
    end_results = pd.DataFrame(columns=pd.MultiIndex.from_product([masking_methodes, 
                                                                   ['train syn', 'val syn', 'test syn', 'train real', 'val real', 'test real']]))

    csv_file = os.path.join(store_path_root, 'best_results.csv')
    for i in range(len(masking_methodes)):
        mask_methode = masking_methodes[i]
        print(mask_methode)

        store_path = Path(os.path.join(store_path_root, f"n2score-{mask_methode}"))
        store_path.mkdir(parents=True, exist_ok=True)
        tmp = Path(os.path.join(store_path, "tensorboard"))
        tmp.mkdir(parents=True, exist_ok=True)
        tmp = Path(os.path.join(store_path, "models"))
        tmp.mkdir(parents=True, exist_ok=True)

        print(f"n2score {i}")
        dataLoader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
        dataLoader_validate = DataLoader(dataset_validate, batch_size=batchsize, shuffle=False)
        dataLoader_test = DataLoader(dataset_test, batch_size=batchsize, shuffle=False)

        #if modi == 0 or modi == 2:
        model = U_Net(1, first_out_chanel=4, scaling_kernel_size=(1,4), conv_kernel=(3,5), batchNorm=batchnorm, n2self_architecture=False).to(device)
        #else:
            #model = Sebastian_N2SUNet(1,1,4).to(device)

        #svae best values in [[train syn, train real], [val syn, val real], [test syn, test real]] structure
        last_loss = [[-1,-1],[-1,-1],[-1,-1]] #lower = better
        best_psnr = [[-1,-1],[-1,-1],[-1,-1]] #higher = better
        best_sv = [[1000,1000],[1000,1000],[1000,1000]] #lower = better
        best_lsd = [[1000,1000],[1000,1000],[1000,1000]] #lower = better (Ruaschsignal ähnlich zur rekonstruction im Frequenzbereich)
        best_coherence = [[-1,-1],[-1,-1],[-1,-1]] #1 = perfekte Korrelation zwischen Rauschsignale und Rekonstruktion; 0 = keine Korrelation
        best_cc = [[-1,-1],[-1,-1],[-1,-1]]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        """
        if modi == 2:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: 0.1 ** (epoch // 100)  # Alle 50 Epochen um einen Faktor von 0.1 reduzieren
            )
        """
        writer = SummaryWriter(log_dir=os.path.join(store_path, "tensorboard"))

        for epoch in tqdm(range(epochs)):
            #break
            #with torch.no_grad():
                #save_example_wave(picture_DAS_syn, model, device, writer, epoch, mask_methode=mask_methode)
            if 'real' in mask_methode:
                break
            loss, psnr, scaledVariance_log, lsd_log, coherence_log, ccGain_log = train(model, device, dataLoader, optimizer, mode="train", writer=writer, epoch=epoch, tweedie=mask_methode)
            writer.add_scalar('Loss Train', statistics.mean(loss), epoch)
            writer.add_scalar('PSNR Train', statistics.mean(psnr), epoch)
            writer.add_scalar('Scaled Variance Train', statistics.mean(scaledVariance_log), epoch)
            writer.add_scalar('LSD Train', statistics.mean(lsd_log), epoch)
            writer.add_scalar('Korelation Train', statistics.mean(coherence_log), epoch)
            writer.add_scalar('cc-Gain Train', statistics.mean(ccGain_log), epoch)

            loss_val, psnr_val, scaledVariance_log_val, lsd_log_val, coherence_log_val, ccGain_log_val = train(model, device, dataLoader_validate, optimizer, mode="val", writer=writer, epoch=epoch, tweedie=mask_methode) 

            #if modi == 2:
                #scheduler.step()

            writer.add_scalar('Loss Val', statistics.mean(loss_val), epoch)
            writer.add_scalar('PSNR Val', statistics.mean(psnr_val), epoch)
            writer.add_scalar('Scaled Variance Val', statistics.mean(scaledVariance_log_val), epoch)
            writer.add_scalar('LSD Val', statistics.mean(lsd_log_val), epoch)
            writer.add_scalar('Korelation Val', statistics.mean(coherence_log_val), epoch)
            writer.add_scalar('cc-Gain Val', statistics.mean(ccGain_log_val), epoch)

            if epoch % 5 == 0  or epoch==epochs-1:
                with torch.no_grad():
                    save_example_wave(picture_DAS_syn, model, device, writer, epoch, mask_methode=mask_methode)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Lernrate', current_lr, epoch)

            if epoch % 50 == 0:
                model_save_path = os.path.join(store_path, "models", f"chk-n2score_{mask_methode}_syn_{epoch}.pth")
                torch.save(model.state_dict(), model_save_path)

            if statistics.mean(psnr) > best_psnr[0][0]:
                best_psnr[0][0] = statistics.mean(psnr)
            if statistics.mean(psnr_val) > best_psnr[1][0]:
                best_psnr[1][0] = statistics.mean(psnr_val)
            if statistics.mean(scaledVariance_log) < best_sv[0][0]:
                best_sv[0][0] = statistics.mean(scaledVariance_log)
            if statistics.mean(scaledVariance_log_val) < best_sv[1][0]:
                best_sv[1][0] = statistics.mean(scaledVariance_log_val)
            #TODO: is lower really better? or is higher better in my case
            if statistics.mean(lsd_log) < best_lsd[0][0]:
                best_lsd[0][0] = statistics.mean(lsd_log)
            if statistics.mean(lsd_log_val) < best_lsd[1][0]:
                best_lsd[1][0] = statistics.mean(lsd_log_val)
            if statistics.mean(coherence_log) > best_coherence[0][0]:
                best_coherence[0][0] = statistics.mean(coherence_log)
            if statistics.mean(coherence_log_val) > best_coherence[1][0]:
                best_coherence[1][0] = statistics.mean(coherence_log_val)
            if statistics.mean(ccGain_log) > best_cc[0][0]:
                best_cc[0][0] = statistics.mean(ccGain_log)
            if statistics.mean(ccGain_log_val) > best_cc[1][0]:
                best_cc[1][0] = statistics.mean(ccGain_log_val)
        #"""
        if 'real' in mask_methode:
            last_loss[0][0] = -1
            last_loss[1][0] = -1
            last_loss[2][0] = -1
            best_psnr[0][0] = -1
            best_psnr[1][0] = -1
            best_sv[0][0] = -1
            best_sv[1][0] = -1
            best_lsd[0][0] = -1
            best_lsd[1][0] = -1
            best_coherence[0][0] = -1
            best_coherence[1][0] = -1
            best_cc[0][0] = -1
            best_cc[1][0] = -1
            best_psnr[2][0] = -1
            best_sv[2][0] = -1
            best_lsd[2][0] = -1
            best_coherence[2][0] = -1
            best_cc[2][0] = -1

        else:
            loss_test, psnr_test, scaledVariance_log_test, lsd_log_test, coherence_log_test, ccGain_log_test = train(model, device, dataLoader_test, optimizer, mode="test", writer=writer, epoch=0, tweedie=mask_methode)
            writer.add_scalar('Loss Test', statistics.mean(loss_test), 0)
            writer.add_scalar('PSNR Test', statistics.mean(psnr_test), 0)
            writer.add_scalar('Scaled Variance Test', statistics.mean(scaledVariance_log_test), 0)
            writer.add_scalar('LSD Test', statistics.mean(lsd_log_test), 0)
            writer.add_scalar('Korelation Test', statistics.mean(coherence_log_test), 0)
            writer.add_scalar('cc-Gain Test', statistics.mean(ccGain_log_test), 0)
            model_save_path = os.path.join(store_path, "models", f"last-model-n2score_{mask_methode}_syn.pth")
            torch.save(model.state_dict(), model_save_path)
            best_psnr[2][0] = statistics.mean(psnr_test)
            best_sv[2][0] = statistics.mean(scaledVariance_log_test)
            best_lsd[2][0] = statistics.mean(lsd_log_test)
            best_coherence[2][0] = statistics.mean(coherence_log_test)
            best_coherence[2][0] = statistics.mean(ccGain_log_test)
            last_loss[0][0] = loss[-1]
            last_loss[1][0] = loss_val[-1]
            last_loss[2][0] = loss_test[-1]
        #"""

        #-------------real data----------------
        #"""
        print("real data")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        dataLoader = DataLoader(real_dataset, batch_size=batchsize, shuffle=True)
        dataLoader_validate = DataLoader(real_dataset_val, batch_size=batchsize, shuffle=False)
        dataLoader_test = DataLoader(real_dataset_test, batch_size=batchsize, shuffle=False)
        for epoch in tqdm(range(realEpochs)):
            #with torch.no_grad():
                #denoised_list, _ ,_ = reconstruct(picture_DAS_real1.unsqueeze(0).unsqueeze(0), model, tweedie=mask_methode)

            loss, psnr, scaledVariance_log, lsd_log, coherence_log, ccGain_log = train(model, device, dataLoader, optimizer, mode="train", writer=writer, epoch=epoch+epochs, tweedie=mask_methode)
            writer.add_scalar('Loss Train', statistics.mean(loss), epoch+epochs)
            writer.add_scalar('PSNR Train', statistics.mean(psnr), epoch+epochs)
            writer.add_scalar('Scaled Variance Train', statistics.mean(scaledVariance_log), epoch+epochs)
            writer.add_scalar('LSD Train', statistics.mean(lsd_log), epoch+epochs)
            writer.add_scalar('Korelation Train', statistics.mean(coherence_log), epoch+epochs)
            writer.add_scalar('cc-Gain Train', statistics.mean(ccGain_log), epoch+epochs)

            loss_val, psnr_val, scaledVariance_log_val, lsd_log_val, coherence_log_val, ccGain_log_val = train(model, device, dataLoader_validate, optimizer, mode="val", writer=writer, epoch=epoch+epochs, tweedie=mask_methode) 

            #if modi == 2:
                #scheduler.step()

            writer.add_scalar('Loss Val', statistics.mean(loss_val), epoch+epochs)
            writer.add_scalar('PSNR Val', statistics.mean(psnr_val), epoch+epochs)
            writer.add_scalar('Scaled Variance Val', statistics.mean(scaledVariance_log_val), epoch+epochs)
            writer.add_scalar('LSD Val', statistics.mean(lsd_log_val), epoch+epochs)
            writer.add_scalar('Korelation Val', statistics.mean(coherence_log_val), epoch+epochs)
            writer.add_scalar('cc-Gain Val', statistics.mean(ccGain_log_val), epoch+epochs)

            if epoch % 10 == 0  or epoch==epochs-1:
                with torch.no_grad():
                    if 'real' in mask_methode:
                        if epoch==(epochs-1):
                            denoised_list, _ ,_ = reconstruct(picture_DAS_real1.unsqueeze(0).unsqueeze(0), model, tweedie=mask_methode)
                            noise_model = ['gaus', 'poisson', 'gamma', 'bernoulli', 'expo']
                            for k, denoised in enumerate(denoised_list):
                                denoised = denoised.cpu().detach().numpy()
                                save_example_wave(picture_DAS_real1, model, device, writer, epoch+epochs, denoised[0][0], vmin=-1, vmax=1, mask_methode=noise_model[k])
                        else:
                            denoised, _, _ = reconstruct(picture_DAS_real1.unsqueeze(0).unsqueeze(0), model, tweedie=mask_methode)
                            denoised = denoised[0].cpu().detach().numpy()
                            save_example_wave(picture_DAS_real1, model, device, writer, epoch+epochs, denoised[0][0], vmin=-1, vmax=1, mask_methode=mask_methode)
                    else:
                        denoised, _, _ = reconstruct(picture_DAS_real1.unsqueeze(0).unsqueeze(0), model, tweedie=mask_methode)
                        denoised = denoised.cpu().detach().numpy()
                        save_example_wave(picture_DAS_real1, model, device, writer, epoch+epochs, denoised[0][0], vmin=-1, vmax=1, mask_methode=mask_methode)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Lernrate', current_lr, epoch+epochs)

            if epoch % 20 == 0:
                model_save_path = os.path.join(store_path, "models", f"chk-n2score_{mask_methode}_real_{epoch}.pth")
                torch.save(model.state_dict(), model_save_path)

            if statistics.mean(psnr) > best_psnr[0][1]:
                best_psnr[0][1] = statistics.mean(psnr)
            if statistics.mean(psnr_val) > best_psnr[1][1]:
                best_psnr[1][1] = statistics.mean(psnr_val)
            if statistics.mean(scaledVariance_log) < best_sv[0][1]:
                best_sv[0][1] = statistics.mean(scaledVariance_log)
            if statistics.mean(scaledVariance_log_val) < best_sv[1][1]:
                best_sv[1][1] = statistics.mean(scaledVariance_log_val)
            #TODO: is lower really better? or is higher better in my case
            if statistics.mean(lsd_log) < best_lsd[0][1]:
                best_lsd[0][1] = statistics.mean(lsd_log)
            if statistics.mean(lsd_log_val) < best_lsd[1][1]:
                best_lsd[1][1] = statistics.mean(lsd_log_val)
            if statistics.mean(coherence_log) > best_coherence[0][1]:
                best_coherence[0][1] = statistics.mean(coherence_log)
            if statistics.mean(coherence_log_val) > best_coherence[1][1]:
                best_coherence[1][1] = statistics.mean(coherence_log_val)
            if statistics.mean(ccGain_log) > best_cc[0][1]:
                best_cc[0][1] = statistics.mean(ccGain_log)
            if statistics.mean(ccGain_log_val) > best_cc[1][1]:
                best_cc[1][1] = statistics.mean(ccGain_log_val)
                
        loss_test, psnr_test, scaledVariance_log_test, lsd_log_test, coherence_log_test, ccGain_log_test = train(model, device, dataLoader_test, optimizer, mode="test", writer=writer, epoch=1, tweedie=mask_methode)
        writer.add_scalar('Loss Test', statistics.mean(loss_test), 1)
        writer.add_scalar('PSNR Test', statistics.mean(psnr_test), 1)
        writer.add_scalar('Scaled Variance Test', statistics.mean(scaledVariance_log_test), 1)
        writer.add_scalar('LSD Test', statistics.mean(lsd_log_test), 1)
        writer.add_scalar('Korelation Test', statistics.mean(coherence_log_test), 1)
        writer.add_scalar('cc-Gain Test', statistics.mean(ccGain_log_test), 1)
        model_save_path = os.path.join(store_path, "models", f"last-model-n2score_{mask_methode}_real.pth")
        torch.save(model.state_dict(), model_save_path)
        best_psnr[2][1] = statistics.mean(psnr_test)
        best_sv[2][1] = statistics.mean(scaledVariance_log_test)
        best_lsd[2][1] = statistics.mean(lsd_log_test)
        best_coherence[2][1] = statistics.mean(coherence_log_test)
        best_cc[2][1] = statistics.mean(ccGain_log_test)
        last_loss[0][1] = loss[-1]
        last_loss[1][1] = loss_val[-1]
        last_loss[2][1] = loss_test[-1]
        #"""

        # Ergebnisse in den DataFrame einfügen
        #'train syn', 'val syn', 'test syn', 'train real', 'val real', 'test real'
        #svae best values in [[train syn, train real], [val syn, val real], [test syn, test real]] structure
        end_results.loc[:, (mask_methode, 'train syn')] = [last_loss[0][0], best_psnr[0][0], best_sv[0][0], best_lsd[0][0], best_coherence[0][0], best_cc[0][0]]
        end_results.loc[:, (mask_methode, 'train real')] = [last_loss[0][1], best_psnr[0][1], best_sv[0][1], best_lsd[0][1], best_coherence[0][1], best_cc[0][1]]

        end_results.loc[:, (mask_methode, 'val syn')] = [last_loss[1][0], best_psnr[1][0], best_sv[1][0], best_lsd[1][0], best_coherence[1][0], best_cc[1][0]]
        end_results.loc[:, (mask_methode, 'val real')] = [last_loss[1][1], best_psnr[1][1], best_sv[1][1], best_lsd[1][1], best_coherence[1][1], best_cc[1][1]]

        end_results.loc[:, (mask_methode, 'test syn')] = [last_loss[2][0], best_psnr[2][0], best_sv[2][0], best_lsd[2][0], best_coherence[2][0], best_cc[2][0]]
        end_results.loc[:, (mask_methode, 'test real')] = [last_loss[2][1], best_psnr[2][1], best_sv[2][1], best_lsd[2][1], best_coherence[2][1], best_cc[2][1]]
        end_results.index = ['Last Loss', 
                         'Best PSNR', 
                         'Best Scaled Variance', 
                         'Best LSD',
                         'Best Coherence',
                         'Best cc-Gain']
        end_results.to_csv(csv_file, index=True)
        modi += 1

    print("n2score fertig")
    #print(str(datetime.now().replace(microsecond=0)).replace(':', '-'))
    return best_psnr, best_sv, best_lsd, best_coherence, best_cc

if __name__ == '__main__':
    main()