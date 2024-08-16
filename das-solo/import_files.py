import os
from pathlib import Path
import shutil
from datetime import datetime
import torch
from torch import nn
#from torch_pconv import PConv2d
import time
import numpy as np
import h5py
from scipy import signal
from torch.utils.data import Dataset


class U_Net(nn.Module):
    def __init__(self, in_chanel = 1, first_out_chanel=64, scaling_kernel_size=2, conv_kernel=3, batchNorm=False, skipLast=True):
        """
        Args:
            in_chanel: Input chanels, default = 1
            first_out_chanel: First Output Chanel Size, default = 64
            scaling_kernel_size: kernel_size and stride for Maxpool2d and upsampling with ConvTranspose2d. Change for DAS to (1,2), defaullt = 2
            conv_kernel: kernel_size for Conv2d, default = 3
            batchhNorm: activate batchnorm in doubleConv-Layer, default = False
            skipLast: should the last skip connection (in the boottom of thhe U-Net pictre) be active, default = True
        """
        super(U_Net, self).__init__()
        
        self.skipLast = skipLast
        self.scaling_kernel_size = scaling_kernel_size
        self.encoder1 = doubleConv(in_chanel, first_out_chanel, conv_kernel=3, norm=batchNorm)
        
        self.encoder2 = nn.Sequential(
            #nn.MaxPool2d(kernel_size=scaling_kernel_size, stride=scaling_kernel_size),
            BlurPool(first_out_chanel, stride=scaling_kernel_size),  # Replace MaxPool2d with BlurPool
            doubleConv(first_out_chanel, first_out_chanel*2, conv_kernel, batchNorm),
        )
        self.encoder3 = nn.Sequential(
            #nn.MaxPool2d(kernel_size=scaling_kernel_size, stride=scaling_kernel_size),
            BlurPool(first_out_chanel*2, stride=scaling_kernel_size),  # Replace MaxPool2d with BlurPool
            doubleConv(first_out_chanel*2, first_out_chanel*4, conv_kernel, batchNorm),
        )
        self.encoder4 = nn.Sequential(
            #nn.MaxPool2d(kernel_size=scaling_kernel_size, stride=scaling_kernel_size),
            BlurPool(first_out_chanel*4, stride=scaling_kernel_size),  # Replace MaxPool2d with BlurPool
            doubleConv(first_out_chanel*4, first_out_chanel*8, conv_kernel, batchNorm),
        )
        self.encoder5 = nn.Sequential(
            #nn.MaxPool2d(kernel_size=scaling_kernel_size, stride=scaling_kernel_size),
            BlurPool(first_out_chanel*8, stride=scaling_kernel_size),  # Replace MaxPool2d with BlurPool
            doubleConv(first_out_chanel*8, first_out_chanel*16, conv_kernel, batchNorm),
        )

        self.decoder1 = Up(first_out_chanel*16, first_out_chanel*8, scaling_kernel_size, conv_kernel, batchNorm, skipConnection=self.skipLast)
        self.decoder2 = Up(first_out_chanel*8, first_out_chanel*4, scaling_kernel_size, conv_kernel, batchNorm)
        self.decoder3 = Up(first_out_chanel*4, first_out_chanel*2, scaling_kernel_size, conv_kernel, batchNorm)
        self.decoder4 = Up(first_out_chanel*2,first_out_chanel, scaling_kernel_size, conv_kernel, batchNorm)
        self.final_conv = nn.Conv2d(first_out_chanel, in_chanel, kernel_size=1)
        self.apply(layer_init)

    def forward(self, x):
        # Encoder
        skip1 = self.encoder1(x)  # (N, 4, 11, 2048)
        skip2 = self.encoder2(skip1)  # (N, 8, 11, 512)
        skip3 = self.encoder3(skip2)  # (N, 16, 11, 128)
        skip4 = self.encoder4(skip3)  # (N, 32, 11, 32)
        result = self.encoder5(skip4)  # (N, 64, 11, 8)
        #print(x.shape)
        #print(skip1.shape)
        #print(skip2.shape) 
        #print(skip3.shape) 
        #print(skip4.shape)
        #print(result.shape)

        # Decoder with Skip Connections
        if self.skipLast:
            result = self.decoder1(result, skip4) # (N, 512, 16, 16  
        else:
            result = self.decoder1(result) # (N, 512, 16, 16)
        result = self.decoder2(result, skip3)  # (N, 256, 32, 32)
        result = self.decoder3(result, skip2)  # (N, 128, 64, 64)
        result = self.decoder4(result, skip1)  # (N, 64, 128, 128)
        
        result = self.final_conv(result)  # (N, 3, 128, 128)
        return result
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channel, out_channel, scaling_kernel_size, conv_kernel, batchNorm, skipConnection=True):
        super().__init__()
        self.scaling_kernel_size = scaling_kernel_size
        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=scaling_kernel_size, stride=scaling_kernel_size)
        self.reduceChanel = nn.Conv2d(in_channel, in_channel//2, kernel_size=1)
        if skipConnection == True:
            self.conv = doubleConv(in_channel, out_channel, conv_kernel, batchNorm)
        else: 
            self.conv = doubleConv(in_channel//2, out_channel, conv_kernel, batchNorm)
        self.apply(layer_init)

    def forward(self, x1, x2=None):
        #x = self.up(x1)
        x = nn.functional.interpolate(x1, scale_factor=self.scaling_kernel_size, mode='bilinear', align_corners=True)
        x = self.reduceChanel(x)
        if x2 is not None:  # skip konnection (immer auser bei N2Same)
            x = torch.cat((x, x2), dim=1)
        return self.conv(x)
    
class doubleConv(nn.Module):
    """
    conv(3x3) -> Batchnorm? -> Relu -> conv(3x3) - Batchnorm? -> Relu
    """
    def __init__(self, in_channels, out_channels, conv_kernel, norm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel, padding="same")
        self.normLayer1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=conv_kernel, padding="same")
        self.normLayer2 = nn.BatchNorm2d(out_channels)
        self.norm = norm
        self.apply(layer_init)

    def forward(self, x):
        x = self.conv1(x)
        if self.norm:
            x = self.normLayer1(x)
        x = self.conv2(nn.functional.relu(x))
        if self.norm:
            x = self.normLayer2(x)
        return nn.functional.relu(x)
    
class BlurPool(nn.Module):
    #for antialiasing downsampling
    def __init__(self, channels, stride=2):
        super(BlurPool, self).__init__()
        self.stride = stride
        self.channels = channels

        # Define a 2D Gaussian kernel
        kernel = torch.tensor([[1., 2., 1.],
                               [2., 4., 2.],
                               [1., 2., 1.]])
        kernel = kernel / kernel.sum()
        kernel = kernel[None, None, :, :].repeat(channels, 1, 1, 1)

        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = nn.functional.pad(x, (1, 1, 1, 1), mode='reflect')
        x = nn.functional.conv2d(x, self.weight, stride=self.stride, groups=self.channels)
        return x

def layer_init(layer, std=0.1, bias_const=0.0):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #nn.init.kaiming_normal_(layer.weight)
        nn.init.orthogonal_(layer.weight)
        nn.init.constant_(layer.bias, 0)


class SyntheticNoiseDAS(Dataset):
    def __init__(self, eq_strain_rates, 
                 nx=11, nt=2048, eq_slowness=(1e-4, 5e-3), log_SNR=(-2,4),
                 gauge=4, fs=50.0, size=1000, mode="train"):
        #size = 1000 bedeutet, insgesammt 1000 Samples auf allen Chaneln
        #       1 Chanel hat ca. 1000/11=90.9 Samples
        #fs = Abtastrate in Hz -> default: 50 Samples in einer Sekunde
        #nt = Zeitfenster: 2048 Samples -> 2 sekunden
        self.eq_strain_rates = eq_strain_rates / eq_strain_rates.std(dim=-1, keepdim=True)
        self.nx = nx
        self.nt = nt
        self.eq_slowness = eq_slowness
        self.log_SNR = log_SNR
        self.gauge = gauge
        self.fs = fs
        self.size = size
        self.mode = mode
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        
        eq_strain_rate = self.eq_strain_rates[np.random.randint(0,len(self.eq_strain_rates))].clone()
        if np.random.random() < 0.5:
            eq_strain_rate = torch.flip(eq_strain_rate, dims=(0,))
        if np.random.random() < 0.5:
            eq_strain_rate *= -1
            
        slowness = np.random.uniform(*self.eq_slowness)
        #slowness = self.eq_slowness
        if np.random.random() < 0.5:
            slowness *= -1
        eq_das = generate_synthetic_das(eq_strain_rate, self.gauge, self.fs, slowness, nx=self.nx)
        j = np.random.randint(0, eq_strain_rate.shape[-1]-self.nt+1)
        eq_das = eq_das[:,j:j+self.nt]
        #"""
        snr = 10 ** np.random.uniform(*self.log_SNR)  # log10-uniform distribution
        amp = 2 * np.sqrt(snr) / torch.abs(eq_das + 1e-10).max() #rescale so, max amplitude = 2*wrt(snr)
        eq_das *= amp

        # 1-10 Hz filtered Gaussian white noise
        gutter = 100
        noise = np.random.randn(self.nx, self.nt + 2*gutter)
        noise = torch.from_numpy(bandpass(noise, 1.0, 10.0, self.fs, gutter).copy())

        sample = eq_das + noise
        scale = sample.std(dim=-1, keepdim=True)
        sample /= scale
        #           nois_image,         clean               noise         std der Daten (b,c,nx,1)  Ampllitude vom DAS (b)
        return sample.unsqueeze(0), eq_das.unsqueeze(0), noise.unsqueeze(0), scale.unsqueeze(0), amp
        #"""
        #return eq_das.unsqueeze(0)

def bandpass(x, low, high, fs, gutter, alpha=0.1):
    """
    alpha: taper length
    """
    
    passband = [2 * low/fs, 2 * high/fs]
    b, a = signal.butter(2, passband, btype="bandpass")
    window = signal.windows.tukey(x.shape[-1], alpha=alpha)
    x = signal.filtfilt(b, a, x * window, axis=-1)

    return x[..., gutter:-gutter]
def generate_synthetic_das(strain_rate, gauge, fs, slowness, nx=512):

    # shift
    # slowness: 0.0001 s/m = 0.1 s/km   -  0.005 s/m = 5 s/km
    # speed: 10,000 m/s = 10 km/s    -  200 m/s = 0.2 km/s
    shift = gauge * fs * slowness # L f / v

    sample = torch.zeros((nx, len(strain_rate)))
    for i in range(nx):
        sample[i] = torch.roll(strain_rate, int(i*shift + np.random.randn(1)))
    
    return sample

def log_files():
    current_path = Path(os.getcwd())
    store_path = Path(os.path.join(current_path, "runs", f"run-{str(datetime.now().replace(microsecond=0)).replace(':', '-')}"))
    store_path.mkdir(parents=True, exist_ok=True)
    models_path = Path(os.path.join(store_path, "models"))
    models_path.mkdir(parents=True, exist_ok=True)
    tensorboard_path = Path(os.path.join(store_path, "tensorboard"))
    tensorboard_path.mkdir(parents=True, exist_ok=True)
    for file in os.listdir(current_path):
        if file.endswith(".py"):
            print("copy: ", file)
            shutil.copyfile(os.path.join(current_path, file), os.path.join(store_path, file))
    print(f"python files are stored. Path: {store_path}")
    return store_path