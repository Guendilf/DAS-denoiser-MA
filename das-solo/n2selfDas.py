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
from skimage.transform import resize

from import_files import gerate_spezific_das, log_files, bandpass, jinv_recon, n2self_mask, mask_random
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

epochs = 300 #2.000 epochen - 1 Epoche = 3424 samples
realEpochs = 100
batchsize = 32
maskChanels = 1
dasChanelsTrain = 11*maskChanels
dasChanelsVal = 11*maskChanels
dasChanelsTest = 11*maskChanels
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
def load_data(strain_train_dir, strain_test_dir, nx=None):
    global dasChanelsTrain
    global dasChanelsVal
    global dasChanelsTest
    if nx:
        dasChanelsTrain = nx
        dasChanelsVal = nx
        dasChanelsTest = nx
        
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
            if channel_spacing == 20:
                DAS_sample = DAS_sample[::5] #if dx = 20
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

    real_dataset = RealDAS(train_real_data, nx=dasChanelsTrain, nt=nt, size=300*batchsize)
    real_dataset_val = RealDAS(val_real_data, nx=dasChanelsVal, nt=nt, size=20*batchsize)
    real_dataset_test = RealDAS(test_real_data, nx=dasChanelsTest, nt=nt, size=20*batchsize)
    #"""
    print("Datensätze geladen!")
    return eq_strain_rates_test,dataset,dataset_validate,dataset_test,test_real_data,real_dataset,real_dataset_val,real_dataset_test

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

def reconstruct(model, device, data, nx=11, nt=2048, batch_size=32, num_masked_channels=1, mask_methode='channel_1'):
    #TODO: make it work with whole batches
    #start = time.time()
    if "channel" in mask_methode:
        data = data.squeeze(1)
        datas = data.split(1, dim=0)
        _, num_masked_channels = mask_methode.split('_')
        num_masked_channels = int(num_masked_channels)
    if "pixel" in mask_methode:
        if "jinv" in mask_methode:
            result = jinv_recon(data, model, grid_size=3, mode='interpolate', include_mask_as_input=False)
            #print(f"reconstruct jinv: {time.time()-start}")
        elif "zero" in mask_methode:
            result = jinv_recon(data, model, grid_size=3, mode='zero', include_mask_as_input=False)
            #print(f"reconstruct zero: {time.time()-start}")
        else:
            result = model(data.to(device))
            #print(f"reconstruct random: {time.time()-start}")
        return result
    recs = []
    for das in datas:
        recs.append(channelwise_reconstruct_part(model, device, das[0], nx, nt, num_masked_channels, mask_methode)) #das.shape=(1,nx,nt)
    #print(f"reconstruct channel: {time.time()-start}")
    return torch.stack(recs).unsqueeze(1).to(device)
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
    data_pad = torch.nn.functional.pad(data, (0,0,lower_res, upper_res), mode='constant') #shape=(311,2048)
    
    masks = torch.ones((NX, 1, nx, nt)).to(device) #shape=(300,1,11,2048)
    #amount of masked channels
    center_idx = nx // 2 #11//2 = 5
    half_mask = num_masked_channels // 2 #3//2 = 1
    start_idx = max(center_idx - half_mask, 0) #max(5-1,0) = 4
    end_idx = min(center_idx + half_mask + (num_masked_channels % 2), nx) #min(5+1+(3%2), 11) = 7
    # num mask chanel = 1 -> 5-6  num mask chanel = 2 -> 4-6  num mask chanel = 3 -> 4-7  num mask chanel = 4 -> 5-6
    masks[:, :, start_idx:end_idx, :] = 0
           
    if ("channel") in mask_methode:
        pass
    elif "circle" in mask_methode:    #machen mit circle und so wie im n2self paper
        _, r = mask_methode.split('_')
        r = int(r)
        if r*2 > nx:
            raise ValueError(f"radius {r} is to big for {nx} channels")
    elif "oval" in mask_methode:
        _, w, h = mask_methode.split('_')
        w = int(w)
        h = int(h)
        r = min(w,h)
        if r*2 > nx:
            raise ValueError(f"radius {r} is to big for {nx} channels")
    else:
        raise ValueError("mask_methode not known")
    #plt.imshow(masks[0][0].detach().cpu().numpy(), origin='lower', interpolation='nearest', cmap='seismic', aspect='auto')
    #plt.show()
    for i in range(num_patches_t): #1
        noisy_samples = torch.zeros((NX, 1, nx, nt)).to(device) #shape=(300,1,11,2048)
        for j in range(NX): #NX = 300
            noisy_samples[j] = data_pad[j:j+nx, i*stride:i*stride + nt] #für j = 299 ist data_pas.shape = (11, 2048)
        if "circle" in mask_methode:
            x = interpolate_disk(noisy_samples, masks, r, num_masked_channels).float().to(device)
        elif "oval" in mask_methode:
            x = interpolate_disk(noisy_samples, masks, r, maskChanels, oval_height=h, oval_width=w).float().to(device)
        elif "random" in mask_methode:
            random_values = torch.normal(0, 0.2, size=masks.shape).to(device) #shape wie noisy_samples
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
        if "circle" in mask_methode:
            x = interpolate_disk(noisy_samples, masks, r, num_masked_channels).float().to(device)
        elif "oval" in mask_methode:
            x = interpolate_disk(noisy_samples, masks, r, maskChanels, oval_height=h, oval_width=w).float().to(device)
        elif "random" in mask_methode:
            random_values = torch.normal(0, 0.2, size=masks.shape).to(device) #shape wie noisy_samples
            x = (noisy_samples * masks + random_values*(1-masks)).float().to(device)
        else:
            x = (noisy_samples * masks).float().to(device)
        out = (model(x) * (1 - masks)).detach().cpu()
        rec[:, -nt:] += torch.sum(out, axis=(1,2))
        freq[:, -nt:] += torch.sum(1 - masks.detach().cpu(), axis=(1,2))
        
    rec /= freq    
    return rec

def interpolate_disk(tensor, masks, radius, width, oval_height=None, oval_width=None):
    """From masks.py of the uppest folder used for pictures"""
    masks = masks.to(tensor.device)
    masks_inv = (1-masks).to(tensor.device)
    
    # Dynamisch den Kernel mit dem gewünschten Radius generieren
    kernel_np = n2self_mask_mit_loch_center(radius, width, oval_height, oval_width)
    kernel_np = kernel_np[np.newaxis, np.newaxis, :, :]  # Zu (1, 1, h, w) formen
    kernel_tensor = torch.Tensor(kernel_np).to(tensor.device)
    kernel_tensor = kernel_tensor / kernel_tensor.sum()

    # Repliziere den Kernel für jeden Farbkanal (c Kanäle)
    kernel_tensor = kernel_tensor.repeat(1, tensor.shape[1], 1, 1)
    filtered_tensor = torch.nn.functional.conv2d(tensor, kernel_tensor, stride=1, padding=kernel_tensor.shape[-1]//2)
    
    # Wende die Maske an: wo mask == 1 wird interpoliert, wo mask_inv == 1 bleibt der originale Wert
    return filtered_tensor * masks + tensor * masks_inv
def n2self_mask_mit_loch_center(radius, width, oval_height, oval_width):
    kernel = disk(radius).astype(np.float32)
    #if oval instead of circle
    if oval_width:
        kernel = resize(kernel, (oval_width, oval_height), mode='reflect', anti_aliasing=True)
        #make it square
        new_size = max(oval_width, oval_height)
        new_size = 2 * new_size - 1 if new_size % 2 == 0 else new_size
        new_array = np.zeros((new_size, new_size), dtype=kernel.dtype)
        start_x = (new_size - oval_width) // 2
        start_y = (new_size - oval_height) // 2
        new_array[start_x:start_x + oval_width, start_y:start_y + oval_height] = kernel
        kernel = new_array

    
    center = len(kernel) // 2
    kernel[center, center] = 0  # Setze das Zentrum auf 0
    if width == 0:
        return kernel
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



def save_example_wave(clean_das_original, model, device, writer, epoch, real_denoised=None, vmin=-1, vmax=1, mask_methode='channel_1'):
    SNRs = [0.1, 1, 10]
    all_noise_das = []
    all_denoised_das = []
    amps = []
    stds = []
    if clean_das_original.shape[0] == 304:
        for SNR in SNRs:
            noise = np.random.randn(*clean_das_original.shape)  # Zufälliges Rauschen
            noise = torch.from_numpy(noise).to(device).float()
            snr = 10 ** SNR
            amp = 2 * np.sqrt(snr) / torch.abs(clean_das_original + 1e-10).max()
            amps.append(amp)
            clean_das = clean_das_original * amp
            noisy_das = clean_das + noise
            noisy_das = noisy_das.unsqueeze(0).unsqueeze(0).to(device).float()
            denoised_waves = reconstruct(model, device, noisy_das/noisy_das.std(), mask_methode=mask_methode, nx=dasChanelsTrain)
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
        all_semblance = semblance(torch.from_numpy(real_denoised).to(device)).squeeze(0).cpu().numpy() #fast implementation without moving window correction
        #all_semblance = moving_window_semblance(np.swapaxes(real_denoised, 0, 1))
        SNRs = ['original']
        channel_idx_1 = 920 // 5 #weil in der bild generierung jeder 5. Kanal überspringen wird
        channel_idx_2 = 3000 // 5
        min_wave = min(clean_das[channel_idx_1].min(),clean_das[channel_idx_2].min())
        max_wave = max(clean_das[channel_idx_1].max(),clean_das[channel_idx_2].max())

        cc_clean = compute_moving_coherence(clean_das, dasChanelsTrain) #11 weil 11 Kanäle in training?
        cc_rec = compute_moving_coherence(real_denoised, dasChanelsTrain) #11 weil 11 Kanäle in training?
        cc_gain_rec = cc_rec / cc_clean
        
        fig = generate_real_das_plot(clean_das, real_denoised, all_semblance, channel_idx_1, channel_idx_2, cc_gain_rec, vmin, vmax, min_wave, max_wave)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = np.array(Image.open(buf))
        writer.add_image(f"Image Plot DAS Real", img, global_step=epoch, dataformats='HWC')
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
    
def channelwise_mask(x, width=1, indices=None):
    """Return:
    mask: mask with 1 for the chosen channels to be blanked out
    """
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
    #start = time.time()
    if "random" in masking_methode:
        masked_noise_image += mask * torch.normal(0, 0.2, size=mask.shape).to(device)
    elif "circle" in masking_methode:    #machen mit circle und so wie im n2self paper
        _, r = masking_methode.split('_')
        r = int(r)
        nx = noise_image.shape[-2]
        if r*2 > nx:
            raise ValueError(f"radius {r} is to big for {nx} channels")
        masked_noise_image = interpolate_disk(noise_image, mask, r, maskChanels).float().to(device) #TODO:can be oval
    elif "oval" in masking_methode:
        _, w, h = masking_methode.split('_')
        w = int(w)
        h = int(h)
        r = min(w,h)
        nx = noise_image.shape[-2]
        if r*2 > nx:
            raise ValueError(f"radius {r} is to big for {nx} channels")
        masked_noise_image = interpolate_disk(noise_image, mask, r, maskChanels, oval_height=h, oval_width=w).float().to(device) #TODO:can be oval
    elif "channel" in masking_methode:
        _, masked_channel_number = masking_methode.split('_')
        masked_channel_number = int(masked_channel_number)
        mask = channelwise_mask(noise_image, width=masked_channel_number)
        masked_noise_image = (1-mask) * noise_image
        #print(f"loss channel: {time.time()-start}")
    elif "pixel" in masking_methode:
        if 'jinv' in masking_methode:
            masked_noise_image, mask = n2self_mask(noise_image, batch_idx, grid_size=3, radius=1) # replace the pixles by the median value of circle with radius
            #print(f"loss jinv: {time.time()-start}")
        elif 'zero' in masking_methode:
            masked_noise_image, mask = n2self_mask(noise_image, batch_idx, grid_size=3, mode='zero') # delete the pixels
            #print(f"loss zero: {time.time()-start}")
        else: #bestimmte anzahl an random pixeln
            _, mask_percent = masking_methode.split('_')
            mask_percent = float(mask_percent)
            mask, _ = mask_random(noise_image, mask_percent, (1,1)) #noise.shape=(batch, 1, nx, nt)
            mask = mask.to(device)
            masked_noise_image = (1-mask) * noise_image
            #print(f"loss random: {time.time()-start}")
    denoised = model(masked_noise_image)
    loss = torch.mean(torch.mean(mask * (denoised - noise_image)**2, dim=-1))
    #print(f"loss calculation: {time.time()-start}")
    #loss_sebastian = torch.mean(torch.sum(mask * (denoised - noise_image)**2, dim=-1)/torch.sum(mask, dim=-1)) #erzeugt nan
    return loss, denoised, mask

def train(model, device, dataLoader, optimizer, mode, writer, epoch, masking_methode='channel_1'):
    global modi
    loss_log = []
    psnr_log = []
    scaledVariance_log = []
    lsd_log = []
    coherence_log = []
    ccGain_log = []
    save_on_last_epoch=True
    for batch_idx, (noise_images, clean, noise, std, amp, _, _, _) in enumerate(dataLoader):
        clean = clean.to(device).type(torch.float32)
        noise_images = noise_images.to(device).type(torch.float32)
        std = std.to(device).type(torch.float32)
        amp = amp.to(device).type(torch.float32)

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
                denoised = reconstruct(model, device, noise_images, mask_methode=masking_methode, nx=dasChanelsTrain)
        #norming
        noise_images *= std
        denoised *= std

        #calculate psnr
        max_intensity=clean.max()-clean.min()
        mse = torch.mean((clean-denoised)**2)
        psnr = 10 * torch.log10((max_intensity ** 2) / mse)
        #calculatte scaled variance (https://figshare.com/articles/software/A_Self-Supervised_Deep_Learning_Approach_for_Blind_Denoising_and_Waveform_Coherence_Enhancement_in_Distributed_Acoustic_Sensing_data/14152277/1?file=26674421) In[13]
        sv = torch.mean((clean - denoised)**2, dim=-1) / torch.mean((clean)**2, dim=-1)
        sv = torch.mean(sv)#, dim=-1)
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
        #cc-gain
        if 'val' in mode or 'test' in mode:
            cc_clean = compute_moving_coherence(clean[0][0].cpu().detach().numpy(), dasChanelsTrain) #11 weil 11 Kanäle in training?
            cc_rec = compute_moving_coherence(denoised[0][0].cpu().detach().numpy(), dasChanelsTrain) #11 weil 11 Kanäle in training?
            cc_value = (np.mean(cc_rec / cc_clean))
        else:
            cc_value = -1
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
    return loss_log, psnr_log, scaledVariance_log, lsd_log, coherence_log, ccGain_log

def main(argv=[]):
    print("Starte Programm n2self!")
    if torch.cuda.device_count() == 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:3"
    
    strain_train_dir = "data/DAS/SIS-rotated_train_50Hz.npy"
    strain_test_dir = "data/DAS/SIS-rotated_test_50Hz.npy"

    #eq_strain_rates_test, dataset, dataset_validate, dataset_test, test_real_data, real_dataset, real_dataset_val, real_dataset_test = load_data(strain_train_dir, strain_test_dir)

    #wave = eq_strain_rates_test[6][4200:6248]
    #picture_DAS_syn = gerate_spezific_das(wave, nx=304, nt=2048, eq_slowness=1/(500), gauge=channel_spacing, fs=sampling)
    #picture_DAS_syn = picture_DAS_syn.to(device).type(torch.float32)
    #picture_DAS_real1 = torch.from_numpy(test_real_data[2][:1472,4576:]).to(device).type(torch.float32) #shape=1482,7424

    
  
    if len(argv) == 1:
        store_path_root = argv[0]
    else:
        store_path_root = log_files()
    global modi
   
    #masking_methodes=['channel_1', 'channel_2', 'channel_3', 'random_value', 'circle_2', 'circle_3', 'oval_2_4', 'oval_3_5', 'pixel_10']
    masking_methodes=['channel_1', 'pixel zero', 'pixel jinv']
    #masking_methodes=['channel_1', 'channel_2', 'channel_3', 'pixel zero', 'pixel jinv', 'channel-m2_1', 'pixel-m2_10', 'pixel jinv-m2']
    masking_methodes=['channel-m2_1', 'pixel-m2_10', 'pixel jinv-m2']
    end_results = pd.DataFrame(columns=pd.MultiIndex.from_product([masking_methodes, 
                                                                   ['train syn', 'val syn', 'test syn', 'train real', 'val real', 'test real']]))
    csv_file = os.path.join(store_path_root, 'best_results.csv')
    load_data_new = 0
    for i in range(len(masking_methodes)):
        mask_methode = masking_methodes[i]
        print(mask_methode)

        store_path = Path(os.path.join(store_path_root, f"n2self-{mask_methode}"))
        store_path.mkdir(parents=True, exist_ok=True)
        tmp = Path(os.path.join(store_path, "tensorboard"))
        tmp.mkdir(parents=True, exist_ok=True)
        tmp = Path(os.path.join(store_path, "models"))
        tmp.mkdir(parents=True, exist_ok=True)

        print(f"n2self {i}")
        

        if 'm2' in mask_methode:
            model = U_Net(1, first_out_chanel=8, scaling_kernel_size=(2,4), conv_kernel=(3,5), batchNorm=batchnorm, n2self_architecture=False).to(device)
            load_data_new += 1
        else:
            model = U_Net(1, first_out_chanel=4, scaling_kernel_size=(1,4), conv_kernel=(3,5), batchNorm=batchnorm, n2self_architecture=False).to(device)
        if load_data_new == 1:
            global dasChanelsTrain
            global dasChanelsVal
            global dasChanelsTest
            dasChanelsTrain = 16
            dasChanelsVal = 16
            dasChanelsTest = 16
            eq_strain_rates_test, dataset, dataset_validate, dataset_test, test_real_data, real_dataset, real_dataset_val, real_dataset_test = load_data(strain_train_dir, strain_test_dir)
            wave = eq_strain_rates_test[6][4200:6248]
            picture_DAS_syn = gerate_spezific_das(wave, nx=304, nt=2048, eq_slowness=1/(500), gauge=channel_spacing, fs=sampling)#304
            picture_DAS_syn = picture_DAS_syn.to(device).type(torch.float32)
            picture_DAS_real1 = torch.from_numpy(test_real_data[2][:1472,4576:]).to(device).type(torch.float32) #shape=1482,7424
        dataLoader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
        dataLoader_validate = DataLoader(dataset_validate, batch_size=batchsize, shuffle=False)
        dataLoader_test = DataLoader(dataset_test, batch_size=batchsize, shuffle=False)

        #svae best values in [[train syn, train real], [val syn, val real], [test syn, test real]] structure
        last_loss = [[-1,-1],[-1,-1],[-1,-1]] #lower = better
        best_psnr = [[-1,-1],[-1,-1],[-1,-1]] #higher = better
        best_sv = [[1000,1000],[1000,1000],[1000,1000]] #lower = better
        best_lsd = [[1000,1000],[1000,1000],[1000,1000]] #lower = better (Ruaschsignal ähnlich zur rekonstruction im Frequenzbereich)
        best_coherence = [[-1,-1],[-1,-1],[-1,-1]] #1 = perfekte Korrelation zwischen Rauschsignale und Rekonstruktion; 0 = keine Korrelation
        best_cc = [[-1,-1],[-1,-1],[-1,-1]] #1 = perfekte Korrelation zwischen Rauschsignale und Rekonstruktion; 0 = keine Korrelation


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
            #break
            loss, psnr, scaledVariance_log, lsd_log, coherence_log, ccGain_log = train(model, device, dataLoader, optimizer, mode="train", writer=writer, epoch=epoch, masking_methode=mask_methode)
            writer.add_scalar('Loss Train', statistics.mean(loss), epoch)
            writer.add_scalar('PSNR Train', statistics.mean(psnr), epoch)
            writer.add_scalar('Scaled Variance Train', statistics.mean(scaledVariance_log), epoch)
            writer.add_scalar('LSD Train', statistics.mean(lsd_log), epoch)
            writer.add_scalar('Korelation Train', statistics.mean(coherence_log), epoch)
            writer.add_scalar('cc-Gain Train', statistics.mean(ccGain_log), epoch)
            #break
            loss_val, psnr_val, scaledVariance_log_val, lsd_log_val, coherence_log_val, ccGain_log_val = train(model, device, dataLoader_validate, optimizer, mode="val", writer=writer, epoch=epoch, masking_methode=mask_methode) 
            
            #if modi == 2:
                #scheduler.step()

            writer.add_scalar('Loss Val', statistics.mean(loss_val), epoch)
            writer.add_scalar('PSNR Val', statistics.mean(psnr_val), epoch)
            writer.add_scalar('Scaled Variance Val', statistics.mean(scaledVariance_log_val), epoch)
            writer.add_scalar('LSD Val', statistics.mean(lsd_log_val), epoch)
            writer.add_scalar('Korelation Val', statistics.mean(coherence_log_val), epoch)
            writer.add_scalar('cc-Gain Val', statistics.mean(ccGain_log_val), epoch)

            if epoch % 10 == 0  or epoch==epochs-1:
                with torch.no_grad():
                    save_example_wave(picture_DAS_syn, model, device, writer, epoch, mask_methode=mask_methode)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Lernrate', current_lr, epoch)

            if epoch % 50 == 0:
                model_save_path = os.path.join(store_path, "models", f"chk-n2self_{mask_methode}_syn_{epoch}.pth")
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
        loss_test, psnr_test, scaledVariance_log_test, lsd_log_test, coherence_log_test, ccGain_log_test = train(model, device, dataLoader_test, optimizer, mode="test", writer=writer, epoch=0, masking_methode=mask_methode)
        writer.add_scalar('Loss Test', statistics.mean(loss_test), 0)
        writer.add_scalar('PSNR Test', statistics.mean(psnr_test), 0)
        writer.add_scalar('Scaled Variance Test', statistics.mean(scaledVariance_log_test), 0)
        writer.add_scalar('LSD Test', statistics.mean(lsd_log_test), 0)
        writer.add_scalar('Korelation Test', statistics.mean(coherence_log_test), 0)
        writer.add_scalar('cc-Gain Test', statistics.mean(ccGain_log_test), 0)
        model_save_path = os.path.join(store_path, "models", f"last-model-n2self_{mask_methode}_syn.pth")
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
                #denoised1 = reconstruct(model, device, picture_DAS_real1.unsqueeze(0).unsqueeze(0), mask_methode=mask_methode, nx=dasChanelsTrain).to('cpu').detach().numpy()

            loss, psnr, scaledVariance_log, lsd_log, coherence_log, ccGain_log = train(model, device, dataLoader, optimizer, mode="train", writer=writer, epoch=epoch+epochs, masking_methode=mask_methode)

            writer.add_scalar('Loss Train', statistics.mean(loss), epoch+epochs)
            writer.add_scalar('PSNR Train', statistics.mean(psnr), epoch+epochs)
            writer.add_scalar('Scaled Variance Train', statistics.mean(scaledVariance_log), epoch+epochs)
            writer.add_scalar('LSD Train', statistics.mean(lsd_log), epoch+epochs)
            writer.add_scalar('Korelation Train', statistics.mean(coherence_log), epoch+epochs)
            writer.add_scalar('cc-Gain Train', statistics.mean(ccGain_log), epoch+epochs)
            #break
            loss_val, psnr_val, scaledVariance_log_val, lsd_log_val, coherence_log_val, ccGain_log_val = train(model, device, dataLoader_validate, optimizer, mode="val", writer=writer, epoch=epoch+epochs, masking_methode=mask_methode) 
            
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
                    denoised1 = reconstruct(model, device, picture_DAS_real1.unsqueeze(0).unsqueeze(0), mask_methode=mask_methode, nx=dasChanelsTrain).to('cpu').detach().numpy()
                    save_example_wave(picture_DAS_real1, model, device, writer, epoch+epochs, denoised1[0][0], vmin=-1, vmax=1, mask_methode=mask_methode)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Lernrate', current_lr, epoch+epochs)

            if epoch % 20 == 0:
                model_save_path = os.path.join(store_path, "models", f"chk-n2self_{mask_methode}_real_{epoch}.pth")
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
        #"""       
        loss_test, psnr_test, scaledVariance_log_test, lsd_log_test, coherence_log_test, ccGain_log_test = train(model, device, dataLoader_test, optimizer, mode="test", writer=writer, epoch=1, masking_methode=mask_methode)
        writer.add_scalar('Loss Test', statistics.mean(loss_test), 1)
        writer.add_scalar('PSNR Test', statistics.mean(psnr_test), 1)
        writer.add_scalar('Scaled Variance Test', statistics.mean(scaledVariance_log_test), 1)
        writer.add_scalar('LSD Test', statistics.mean(lsd_log_test), 1)
        writer.add_scalar('Korelation Test', statistics.mean(coherence_log_test), 1)
        writer.add_scalar('cc-Gain Test', statistics.mean(ccGain_log_test), 1)
        model_save_path = os.path.join(store_path, "models", f"last-model-n2self_{mask_methode}_real.pth")
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
        end_results.loc[:, (mask_methode, 'train syn')] = [round(last_loss[0][0],3), round(best_psnr[0][0],3), round(best_sv[0][0],3), round(best_lsd[0][0],3), round(best_coherence[0][0],3), round(best_cc[0][0],3)]
        end_results.loc[:, (mask_methode, 'train real')] = [round(last_loss[0][1],3), round(best_psnr[0][1],3), round(best_sv[0][1],3), round(best_lsd[0][1],3), round(best_coherence[0][1],3), round(best_cc[0][1],3)]

        end_results.loc[:, (mask_methode, 'val syn')] = [round(last_loss[1][0],3), round(best_psnr[1][0],3), round(best_sv[1][0],3), round(best_lsd[1][0],3), round(best_coherence[1][0],3), round(best_cc[1][0],3)]
        end_results.loc[:, (mask_methode, 'val real')] = [round(last_loss[1][1],3), round(best_psnr[1][1],3), round(best_sv[1][1],3), round(best_lsd[1][1],3), round(best_coherence[1][1],3), round(best_cc[1][1],3)]

        end_results.loc[:, (mask_methode, 'test syn')] = [round(last_loss[2][0],3), round(best_psnr[2][0],3), round(best_sv[2][0],3), round(best_lsd[2][0],3), round(best_coherence[2][0],3), round(best_cc[2][0],3)]
        end_results.loc[:, (mask_methode, 'test real')] = [round(last_loss[2][1],3), round(best_psnr[2][1],3), round(best_sv[2][1],3), round(best_lsd[2][1],3), round(best_coherence[2][1],3), round(best_cc[2][1],3)]
        end_results.index = ['Last Loss', 
                         'Best PSNR', 
                         'Best Scaled Variance', 
                         'Best LSD',
                         'Best Coherence',
                         'Best cc-Gain']
        end_results.to_csv(csv_file, index=True)
        modi += 1

    print("n2self fertig")
    return best_psnr, best_sv, best_lsd, best_coherence, best_cc

if __name__ == '__main__':
    main()