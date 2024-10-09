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
from skimage.morphology import disk

from import_files import gerate_spezific_das, log_files, bandpass
from import_files import U_Net, Sebastian_N2SUNet
from unet_copy import UNet as unet
from import_files import SyntheticNoiseDAS, RealDAS
from import_files import semblance
from plots import generate_wave_plot, generate_das_plot3, generate_real_das_plot

#tensorboard --logdir="E:\Bibiotheken\Dokumente\02 Uni\1 MA\Code\DAS-denoiser-MA\runs"      #PC
#tensorboard --logdir="E:\Bibiotheken\Dokumente\02 Uni\1 MA\runs"                           #vom Server
#tensorboard --logdir="C:\Users\LaAlexPi\Documents\01_Uni\MA\runs\"                         #Laptop

epochs = 300 #2.000 epochen - 1 Epoche = 3424 samples
realEpochs = 200
batchsize = 32
maskChanels = 1
dasChanelsTrain = 11*maskChanels
dasChanelsVal = 11*maskChanels
dasChanelsTest = 11*maskChanels
nt = 2048

lr = 0.0001
batchnorm = False
save_model = False

gauge_length = 4#19.2
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
def reconstruct_old(model, device, noise_images):
    buffer = torch.zeros_like(noise_images).to(device)
    training_size = dasChanelsTrain # 11 how much chanels were used for training
    num_chunks = noise_images.shape[2] // training_size
    left_chunks = noise_images.shape[2] % training_size
    for chunk_idx in range(num_chunks):
        # Extract the chunk from the larger picture
        start_idx = chunk_idx * training_size
        chunk = noise_images[:, :, start_idx:start_idx+training_size, :]
        for i in range(training_size):
            """
            mask = torch.zeros_like(chunk).to(device)
            mask[:, :, i, :] = 1  # Mask out the i-th channel
            """
            mask = channelwise_mask(chunk, width=maskChanels, indices=np.full(chunk.shape[0], i))
            input_image = chunk * (1 - mask)
            j_denoised = model(input_image)
            buffer[:, :, start_idx:start_idx+training_size, :] += j_denoised * mask
    # calculate the left overs when chanels are not a multiplicative of training_size
    for i in range(left_chunks):
        chunk = noise_images[:, :, noise_images.shape[2]-dasChanelsTrain:, :]
        mask = torch.zeros_like(chunk).to(device)
        mask[:, :, training_size-i-1, :] = 1
        input_image = chunk * (1 - mask)
        j_denoised = model(input_image)
        buffer[:, :, noise_images.shape[2]-dasChanelsTrain:, :] += j_denoised * mask
    return buffer

def reconstruct(model, device, data, nx=11, nt=2048, batch_size=32, num_masked_channels=1, mask_methode='original'):
    buffer = []
    for sample in data:
        sample = sample.squeeze(0)
        datas = sample.split(batch_size, dim=0)
        recs = []
        for das in datas:
            recs.append(channelwise_reconstruct_part(model, device, das, nx, nt, num_masked_channels, mask_methode))
        buffer.append(torch.cat(recs, dim=0).unsqueeze(0))
    return torch.stack(buffer).to(device)

def channelwise_reconstruct_part_original_sebastian(model, device, data, nx, nt): # nx=11, nt=2048 
    NX, NT = data.shape
    stride = 2048

    NT_pad = (NT // stride) * stride + stride - NT
    num_patches_t = (NT - 2048) // stride + 1
    rec = torch.zeros((NX, NT))
    freq = torch.zeros((NX, NT))
    
    lower_res = int(np.floor(nx/2))
    upper_res = int(np.ceil(nx/2))
    data_pad = torch.nn.functional.pad(data, (0,0,lower_res, upper_res), mode='constant')
    
    masks = torch.ones((NX, 1, nx, nt)).to(device)
    masks[:,:,nx//2] = 0
    for i in range(num_patches_t):    
        noisy_samples = torch.zeros((NX, 1, nx, nt)).to(device)
        for j in range(NX):
            noisy_samples[j] = data_pad[j:j+nx, i*stride:i*stride + nt]
        x = (noisy_samples * masks).float().to(device)
        out = (model(x) * (1 - masks)).detach().cpu()
        rec[:, i*stride:i*stride + nt] += torch.sum(out, axis=(1,2))
        freq[:, i*stride:i*stride + nt] += torch.sum(1 - masks.detach().cpu(), axis=(1,2))

    if NT % stride != 0:
        noisy_samples = torch.zeros((NX, 1, nx, nt)).to(device)
        for j in range(NX):
            noisy_samples[j] = data_pad[j:j+nx, -nt:]
        x = (noisy_samples * masks).float().to(device)
        out = (model(x) * (1 - masks)).detach().cpu()
        rec[:, -nt:] += torch.sum(out, axis=(1,2))
        freq[:, -nt:] += torch.sum(1 - masks.detach().cpu(), axis=(1,2))
        
    rec /= freq    
    return rec

def channelwise_reconstruct_part(model, device, data, nx, nt, num_masked_channels, mask_methode): # nx=11, nt=2048
    """Args:
    model: Model to use for reconstruction
    device: Device to use for computation (gpu or cpu)
    data: noisy DAS data
    nx: channels used while training
    nt: time samples used while training
    num_masked_channels: Number of channels to mask out (default 1)
    mask_methode: Methode to mask out (default 'original')
        original: Masked chanel set to 0, like in the original version of n2self for DAS
        same: Masked channel set to random value
        self_r: Masked channel set to medium of radius r neighbours-channels
    """
    NX, NT = data.shape
    stride = 2048

    NT_pad = (NT // stride) * stride + stride - NT
    num_patches_t = (NT - 2048) // stride + 1
    rec = torch.zeros((NX, NT))
    freq = torch.zeros((NX, NT))
    
    lower_res = int(np.floor(nx/2))
    upper_res = int(np.ceil(nx/2))
    data_pad = torch.nn.functional.pad(data, (0,0,lower_res, upper_res), mode='constant')
    
    masks = torch.ones((NX, 1, nx, nt)).to(device)
    #amount of masked channels
    center_idx = nx // 2 #11//2 = 5
    half_mask = num_masked_channels // 2 #3//2 = 1
    start_idx = max(center_idx - half_mask, 0) #max(5-1,0) = 4
    end_idx = min(center_idx + half_mask + (num_masked_channels % 2), nx) #min(5+1+(3%2), 11) = 7
    # num mask chanel = 1 -> 5-6  num mask chanel = 2 -> 4-6  num mask chanel = 3 -> 4-7  num mask chanel = 4 -> 5-6
    if "original" in mask_methode:
        masks[:, :, start_idx:end_idx, :] = 0
    elif "same" in mask_methode:
        masks[:, :, start_idx:end_idx, :] = 0
        random_values = torch.normal(0, 0.2, size=masks.shape).to(device) #shape wie noisy_samples
    elif "self" in mask_methode:    #machen mit circle und so wie im n2self paper
        _, r = mask_methode.split('_')
        r = int(r)
        if r*2 > nx:
            raise ValueError(f"radius {r} is to big for {nx} channels")
        masks[:, :, start_idx:end_idx, :] = 0
    else:
        raise ValueError("mask_methode not known")
    #plt.imshow(masks[0][0].detach().cpu().numpy(), origin='lower', interpolation='nearest', cmap='seismic', aspect='auto')
    #plt.show()
    for i in range(num_patches_t):    
        noisy_samples = torch.zeros((NX, 1, nx, nt)).to(device)
        for j in range(NX):
            noisy_samples[j] = data_pad[j:j+nx, i*stride:i*stride + nt]
        if "self" in mask_methode:
            x = interpolate_disk(noisy_samples, masks, r, num_masked_channels).float().to(device)
        elif "same" in mask_methode:
            x = (noisy_samples * masks + random_values*(1-masks)).float().to(device)
        else:
            x = (noisy_samples * masks).float().to(device)
        #plt.imshow(x[0][0].detach().cpu().numpy(), origin='lower', interpolation='nearest', cmap='seismic', aspect='auto')
        #plt.show()
        out = (model(x) * (1 - masks)).detach().cpu()
        rec[:, i*stride:i*stride + nt] += torch.sum(out, axis=(1,2))
        freq[:, i*stride:i*stride + nt] += torch.sum(1 - masks.detach().cpu(), axis=(1,2))

    if NT % stride != 0:
        noisy_samples = torch.zeros((NX, 1, nx, nt)).to(device)
        for j in range(NX):
            noisy_samples[j] = data_pad[j:j+nx, -nt:]
        if "self" in mask_methode:
            x = interpolate_disk(noisy_samples, masks, r, num_masked_channels).float().to(device)
        elif "same" in mask_methode:
            x = (noisy_samples * masks + random_values*(1-masks)).float().to(device)
        else:
            x = (noisy_samples * masks).float().to(device)
        out = (model(x) * (1 - masks)).detach().cpu()
        rec[:, -nt:] += torch.sum(out, axis=(1,2))
        freq[:, -nt:] += torch.sum(1 - masks.detach().cpu(), axis=(1,2))
        
    rec /= freq    
    return rec

def interpolate_disk(tensor, masks, radius, width):
    """From masks.py of the uppest folder used for pictures"""
    masks = masks.to(tensor.device)
    masks_inv = (1-masks).to(tensor.device)
    
    # Dynamisch den Kernel mit dem gewünschten Radius generieren
    kernel_np = n2self_mask_mit_loch_center(radius, width)
    kernel_np = kernel_np[np.newaxis, np.newaxis, :, :]  # Zu (1, 1, h, w) formen
    kernel_tensor = torch.Tensor(kernel_np).to(tensor.device)
    kernel_tensor = kernel_tensor / kernel_tensor.sum()

    # Repliziere den Kernel für jeden Farbkanal (c Kanäle)
    kernel_tensor = kernel_tensor.repeat(1, tensor.shape[1], 1, 1)
    filtered_tensor = torch.nn.functional.conv2d(tensor, kernel_tensor, stride=1, padding=radius)
    
    # Wende die Maske an: wo mask == 1 wird interpoliert, wo mask_inv == 1 bleibt der originale Wert
    return filtered_tensor * masks_inv + tensor * masks
def n2self_mask_mit_loch_center(radius, width):
    kernel = disk(radius).astype(np.float32)
    center = len(kernel) // 2
    kernel[center, center] = 0  # Setze das Zentrum auf 0
    kernel[center, :] = 0  # Setze den mittleren Chanel auf 0
    #Maksiere weiter Kanäle, wenn nicht nur der eine mittlere Kanal maskiert werden soll
    channel_masked = 1
    channel_to_mask = width - channel_masked
    while channel_to_mask > 0:
        if channel_masked % 2 == 0: #gerade
            #mask nach unten
            kernel[center+channel_masked/2,:] = 0
        else: #ungerade
            #mask nach oben
            kernel[center-(channel_masked//2+1),:] = 0
        channel_to_mask -= 1
    return kernel

def save_example_wave(clean_das_original, model, device, writer, epoch, real_denoised=None, vmin=-1, vmax=1, mask_methode='original'):
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
            denoised_waves = reconstruct(model, device, noisy_das/noisy_das.std(), mask_methode=mask_methode)
            all_noise_das.append(noisy_das.squeeze(0).squeeze(0))
            all_denoised_das.append((denoised_waves*noisy_das.std()).squeeze(0).squeeze(0))
        
        all_noise_das = torch.stack(all_noise_das)
        all_denoised_das = torch.stack(all_denoised_das)
        amps = torch.stack(amps)
        all_semblance = semblance(all_denoised_das.unsqueeze(1).to(device))
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
    else:
        clean_das = clean_das_original.detach().cpu().numpy()
        all_noise_das = clean_das_original.detach().cpu().numpy()
        all_semblance = semblance(torch.from_numpy(real_denoised).to(device)).squeeze(0).cpu().numpy()
        SNRs = ['original']
        channel_idx_1 = 920 // 5 #weil in der bild generierung jeder 5. Kanal überspringen wird
        channel_idx_2 = 3000 // 5
        min_wave = min(clean_das[channel_idx_1].min(),clean_das[channel_idx_2].min())
        max_wave = max(clean_das[channel_idx_1].max(),clean_das[channel_idx_2].max())
        
        fig = generate_real_das_plot(clean_das, real_denoised, all_semblance, channel_idx_1, channel_idx_2, vmin, vmax, min_wave, max_wave)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = np.array(Image.open(buf))
        writer.add_image(f"Image Plot DAS Real vmin={vmin}", img, global_step=epoch, dataformats='HWC')
        buf.close()

        clean_das = torch.from_numpy(clean_das)
        all_noise_das = torch.from_numpy(all_noise_das)
        all_denoised_das = torch.from_numpy(real_denoised)



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
    
def channelwise_mask(x, width=1, indices=None):

    batch_size, _, nx, nt = x.shape
    mask = torch.ones_like(x)
    u = int(np.floor(width/2))
    l = int(np.ceil(width/2))
    if indices is None:
        indices = torch.randint(u, nx - l, (batch_size,))
    for i in range(batch_size):
        mask[i, :, indices[i]-u:indices[i]+l] = 0
    
    return (1-mask)

def calculate_loss(noise_image, model, batch_idx, masking_methode):
    #masked_noise_image, mask = Mask.n2self_mask(noise_image, batch_idx)
    """
    mask = torch.zeros_like(noise_image).to(noise_image.device)
    for i in range(mask.shape[0]):
        mask[i, :, np.random.randint(0, mask.shape[2]), :] = 1
    """
    device = noise_image.device
    mask = channelwise_mask(noise_image, width=maskChanels)
    masked_noise_image = (1-mask) * noise_image
    if "same" in masking_methode:
        masked_noise_image += mask * torch.normal(0, 0.2, size=mask.shape).to(device)
    elif "self" in masking_methode:    #machen mit circle und so wie im n2self paper
        _, r = masking_methode.split('_')
        r = int(r)
        nx = noise_image.shape[-2]
        if r*2 > nx:
            raise ValueError(f"radius {r} is to big for {nx} channels")
        masked_noise_image = interpolate_disk(noise_image, mask, r, maskChanels).float().to(device)
    denoised = model(masked_noise_image)
    #loss_my = torch.nn.MSELoss()(denoised*(mask), noise_image*(mask))
    return torch.mean(torch.mean(mask * (denoised - noise_image)**2, dim=-1)), denoised, mask

def train(model, device, dataLoader, optimizer, mode, writer, epoch, masking_methode='original'):
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

        #reconstruct(model, device, noise_images, num_masked_channels=1, mask_methode='original')
        #reconstruct(model, device, noise_images, num_masked_channels=1, mask_methode='n2same')
        #reconstruct(model, device, noise_images, num_masked_channels=1, mask_methode='self_1')
        #reconstruct(model, device, noise_images, num_masked_channels=1, mask_methode='self_2')
        #reconstruct(model, device, noise_images, num_masked_channels=1, mask_methode='self_3')
        #reconstruct(model, device, noise_images, num_masked_channels=2, mask_methode='original')
        #reconstruct(model, device, noise_images, num_masked_channels=2, mask_methode='self_2')

        if mode == "train":
            model.train()
            loss, denoised, mask_orig = calculate_loss(noise_images, model, batch_idx, masking_methode)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model.eval()
                loss, denoised, mask_orig = calculate_loss(noise_images, model, batch_idx, masking_methode)
                #J-Invariant reconstruction
                denoised = reconstruct(model, device, noise_images, mask_methode=masking_methode)
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
        if epoch < epochs:
            writer.add_scalar(f'Sigma {mode}', noise.std(), global_step=epoch * len(dataLoader) + batch_idx)
    return loss_log, psnr_log, scaledVariance_log, lsd_log, coherence_log

def main(argv=[]):
    print("Starte Programm n2self!")
    if torch.cuda.device_count() == 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:3"
    
    strain_train_dir = "data/DAS/SIS-rotated_train_50Hz.npy"
    strain_test_dir = "data/DAS/SIS-rotated_test_50Hz.npy"

    print("lade Datensätze ...")
    eq_strain_rates = np.load(strain_train_dir)
    # Normalise waveforms
    """
    N_ch, N_t = eq_strain_rates.shape
    t_slice = slice(N_t//4, 3*N_t//4)
    scaled_data = np.zeros_like(eq_strain_rates)
    for i, wv in enumerate(eq_strain_rates):
        scaled_data[i] = wv / wv[t_slice].std()
    eq_strain_rates = scaled_data
    """
    N_ch, N_t = eq_strain_rates.shape
    t_slice = slice(N_t//4, 3*N_t//4)
    stds = eq_strain_rates[:, t_slice].std(axis=1, keepdims=True)
    eq_strain_rates = eq_strain_rates / stds

    split_idx = int(0.8 * len(eq_strain_rates))
    eq_strain_rates_train = torch.tensor(eq_strain_rates[:split_idx])
    eq_strain_rates_val = torch.tensor(eq_strain_rates[split_idx:])

    eq_strain_rates_test = np.load(strain_test_dir)
    # Normalise waveforms
    """code aus n2self paper
    N_ch, N_t = eq_strain_rates_test.shape
    t_slice = slice(N_t//4, 3*N_t//4)
    scaled_data = np.zeros_like(eq_strain_rates_test)
    for i, wv in enumerate(eq_strain_rates_test):
        scaled_data[i] = wv / wv[t_slice].std()
    eq_strain_rates_test = scaled_data
    """
    N_ch, N_t = eq_strain_rates_test.shape
    t_slice = slice(N_t//4, 3*N_t//4)
    stds = eq_strain_rates_test[:, t_slice].std(axis=1, keepdims=True)
    eq_strain_rates_test = eq_strain_rates_test / stds



    eq_strain_rates_test = torch.tensor(eq_strain_rates_test)
    dataset = SyntheticNoiseDAS(eq_strain_rates_train, nx=dasChanelsTrain, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=300*batchsize)
    dataset_validate = SyntheticNoiseDAS(eq_strain_rates_val, nx=dasChanelsVal, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=30*batchsize)
    dataset_test = SyntheticNoiseDAS(eq_strain_rates_test, nx=dasChanelsTest, eq_slowness=slowness, log_SNR=snr, gauge=gauge_length, size=30*batchsize)


    train_path = "Server_DAS/real_train/"
    test_path = "Server_DAS/real_test/"
    train_path = sorted([train_path + f for f in os.listdir(train_path)])
    test_paths = sorted([test_path + f for f in os.listdir(test_path)])

    train_real_data = []
    for i, p in enumerate(train_path):
        with h5py.File(p, 'r') as hf:
            DAS_sample = hf['DAS'][81:]
            DAS_sample = DAS_sample[::5] #if dx = 20
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
            DAS_sample = DAS_sample[::5] #if dx = 20
            test_real_data.append(DAS_sample)
    test_real_data = np.stack(test_real_data)
    original__real_test_data = test_real_data.copy()
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

    real_dataset = RealDAS(train_real_data, nx=dasChanelsTrain, nt=nt, size=300*batchsize)
    real_dataset_val = RealDAS(val_real_data, nx=dasChanelsVal, nt=nt, size=20*batchsize)
    real_dataset_test = RealDAS(test_real_data, nx=dasChanelsTest, nt=nt, size=20*batchsize)

    print("Datensätze geladen!")

    wave = eq_strain_rates_test[6][4200:6248]
    picture_DAS_syn = gerate_spezific_das(wave, nx=300, nt=2048, eq_slowness=1/(500), gauge=gauge_length, fs=sampling)
    picture_DAS_syn = picture_DAS_syn.to(device).type(torch.float32)
    picture_DAS_real1 = torch.from_numpy(test_real_data[2][:,:4500]).to(device).type(torch.float32)
    picture_DAS_real2 = torch.from_numpy(original__real_test_data[2][:,:4500]).to(device).type(torch.float32) #mit plot von: vmin=-100, vmax=100

    
  
    if len(argv) == 1:
        store_path_root = argv[0]
    else:
        store_path_root = log_files()
    global modi
    #svae best values in [[train syn, train real], [val syn, val real], [test syn, test real]] structure
    best_psnr = [[-1,-1],[-1,-1],[-1,-1]] #higher = better
    best_sv = [[-1,-1],[-1,-1],[-1,-1]] #lower = better
    best_lsd = [[-1,-1],[-1,-1],[-1,-1]] #lower = better (Ruaschsignal ähnlich zur rekonstruction im Frequenzbereich)
    best_coherence = [[-1,-1],[-1,-1],[-1,-1]] #1 = perfekte Korrelation zwischen Rauschsignale und Rekonstruktion; 0 = keine Korrelation

    masking_methodes=['original', 'same', 'self_2', 'self_3']
    for i in range(4):
        mask_methode = masking_methodes[i]
        print(mask_methode)

        store_path = Path(os.path.join(store_path_root, f"n2self-{modi}"))
        store_path.mkdir(parents=True, exist_ok=True)
        tmp = Path(os.path.join(store_path, "tensorboard"))
        tmp.mkdir(parents=True, exist_ok=True)
        tmp = Path(os.path.join(store_path, "models"))
        tmp.mkdir(parents=True, exist_ok=True)

        print(f"n2self {i}")
        dataLoader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
        dataLoader_validate = DataLoader(dataset_validate, batch_size=batchsize, shuffle=False)
        dataLoader_test = DataLoader(dataset_test, batch_size=batchsize, shuffle=False)

        #if modi == 0 or modi == 2:
        model = U_Net(1, first_out_chanel=4, scaling_kernel_size=(1,4), conv_kernel=(3,5), batchNorm=batchnorm, n2self_architecture=False).to(device)
        #else:
            #model = Sebastian_N2SUNet(1,1,4).to(device)

        
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
            #save_example_wave(picture_DAS_syn, model, device, writer, epoch)
            loss, psnr, scaledVariance_log, lsd_log, coherence_log = train(model, device, dataLoader, optimizer, mode="train", writer=writer, epoch=epoch, masking_methode=mask_methode)

            writer.add_scalar('Loss Train', statistics.mean(loss), epoch)
            writer.add_scalar('PSNR Train', statistics.mean(psnr), epoch)
            writer.add_scalar('Scaled Variance Train', statistics.mean(scaledVariance_log), epoch)
            writer.add_scalar('LSD Train', statistics.mean(lsd_log), epoch)
            writer.add_scalar('Korelation Train', statistics.mean(coherence_log), epoch)

            loss_val, psnr_val, scaledVariance_log_val, lsd_log_val, coherence_log_val = train(model, device, dataLoader_validate, optimizer, mode="val", writer=writer, epoch=epoch, masking_methode=mask_methode) 

            #if modi == 2:
                #scheduler.step()

            writer.add_scalar('Loss Val', statistics.mean(loss_val), epoch)
            writer.add_scalar('PSNR Val', statistics.mean(psnr_val), epoch)
            writer.add_scalar('Scaled Variance Val', statistics.mean(scaledVariance_log_val), epoch)
            writer.add_scalar('LSD Val', statistics.mean(lsd_log_val), epoch)
            writer.add_scalar('Korelation Val', statistics.mean(coherence_log_val), epoch)

            if epoch % 5 == 0  or epoch==epochs-1:
                save_example_wave(picture_DAS_syn, model, device, writer, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Lernrate', current_lr, epoch)

            if epoch % 50 == 0:
                model_save_path = os.path.join(store_path, "models", f"chk-n2self{modi}_syn_{epoch}.pth")
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
                
        loss_test, psnr_test, scaledVariance_log_test, lsd_log_test, coherence_log_test = train(model, device, dataLoader_test, optimizer, mode="test", writer=writer, epoch=0, masking_methode=mask_methode)
        writer.add_scalar('Loss Test', statistics.mean(loss_test), 0)
        writer.add_scalar('PSNR Test', statistics.mean(psnr_test), 0)
        writer.add_scalar('Scaled Variance Test', statistics.mean(scaledVariance_log_test), 0)
        writer.add_scalar('LSD Test', statistics.mean(lsd_log_test), 0)
        writer.add_scalar('Korelation Test', statistics.mean(coherence_log_test), 0)
        model_save_path = os.path.join(store_path, "models", f"last-model-n2self{modi}_syn.pth")
        torch.save(model.state_dict(), model_save_path)
        best_psnr[2][0] = statistics.mean(psnr_test)
        best_sv[2][0] = statistics.mean(scaledVariance_log_test)
        best_lsd[2][0] = statistics.mean(lsd_log_test)
        best_coherence[2][0] = statistics.mean(coherence_log_test)
        

        #-------------real data----------------

        print("real data")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        dataLoader = DataLoader(real_dataset, batch_size=batchsize, shuffle=True)
        dataLoader_validate = DataLoader(real_dataset_val, batch_size=batchsize, shuffle=False)
        dataLoader_test = DataLoader(real_dataset_test, batch_size=batchsize, shuffle=False)
        for epoch in tqdm(range(realEpochs)):
            denoised1 = reconstruct(model, device, picture_DAS_real2.unsqueeze(0).unsqueeze(0), mask_methode=mask_methode).to('cpu').detach().numpy()
            save_example_wave(picture_DAS_real1, model, device, writer, epoch+epochs, denoised1[0][0], vmin=-100, vmax=100)

            loss, psnr, scaledVariance_log, lsd_log, coherence_log = train(model, device, dataLoader, optimizer, mode="train", writer=writer, epoch=epoch+epochs, masking_methode=mask_methode)

            writer.add_scalar('Loss Train', statistics.mean(loss), epoch+epochs)
            writer.add_scalar('PSNR Train', statistics.mean(psnr), epoch+epochs)
            writer.add_scalar('Scaled Variance Train', statistics.mean(scaledVariance_log), epoch+epochs)
            writer.add_scalar('LSD Train', statistics.mean(lsd_log), epoch+epochs)
            writer.add_scalar('Korelation Train', statistics.mean(coherence_log), epoch+epochs)

            loss_val, psnr_val, scaledVariance_log_val, lsd_log_val, coherence_log_val = train(model, device, dataLoader_validate, optimizer, mode="val", writer=writer, epoch=epoch+epochs, masking_methode=mask_methode) 

            #if modi == 2:
                #scheduler.step()

            writer.add_scalar('Loss Val', statistics.mean(loss_val), epoch+epochs)
            writer.add_scalar('PSNR Val', statistics.mean(psnr_val), epoch+epochs)
            writer.add_scalar('Scaled Variance Val', statistics.mean(scaledVariance_log_val), epoch+epochs)
            writer.add_scalar('LSD Val', statistics.mean(lsd_log_val), epoch+epochs)
            writer.add_scalar('Korelation Val', statistics.mean(coherence_log_val), epoch+epochs)

            if epoch % 5 == 0  or epoch==epochs-1:
                denoised1 = reconstruct(model, device, picture_DAS_real1.unsqueeze(0).unsqueeze(0), mask_methode=mask_methode).to('cpu').detach().numpy()
                denoised2 = reconstruct(model, device, picture_DAS_real2.unsqueeze(0).unsqueeze(0), mask_methode=mask_methode).to('cpu').detach().numpy()
                save_example_wave(picture_DAS_real1, model, device, writer, epoch+epochs, denoised1[0][0], vmin=-100, vmax=100)
                save_example_wave(picture_DAS_real2, model, device, writer, epoch+epochs, denoised2[0][0], vmin=-100, vmax=100)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Lernrate', current_lr, epoch+epochs)

            if epoch % 50 == 0:
                model_save_path = os.path.join(store_path, "models", f"chk-n2self{modi}_real_{epoch}.pth")
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
                
        loss_test, psnr_test, scaledVariance_log_test, lsd_log_test, coherence_log_test = train(model, device, dataLoader_test, optimizer, mode="test", writer=writer, epoch=1, masking_methode=mask_methode)
        writer.add_scalar('Loss Test', statistics.mean(loss_test), 1)
        writer.add_scalar('PSNR Test', statistics.mean(psnr_test), 1)
        writer.add_scalar('Scaled Variance Test', statistics.mean(scaledVariance_log_test), 1)
        writer.add_scalar('LSD Test', statistics.mean(lsd_log_test), 1)
        writer.add_scalar('Korelation Test', statistics.mean(coherence_log_test), 1)
        model_save_path = os.path.join(store_path, "models", f"last-model-n2self{modi}_real.pth")
        torch.save(model.state_dict(), model_save_path)
        best_psnr[2][1] = statistics.mean(psnr_test)
        best_sv[2][1] = statistics.mean(scaledVariance_log_test)
        best_lsd[2][1] = statistics.mean(lsd_log_test)
        best_coherence[2][1] = statistics.mean(coherence_log_test)
        modi += 1

    print("n2self fertig")
    return best_psnr, best_sv, best_lsd, best_coherence

if __name__ == '__main__':
    main()