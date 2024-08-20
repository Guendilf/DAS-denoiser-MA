import os
from pathlib import Path
import shutil
from datetime import datetime
from matplotlib import pyplot as plt
import torch
from torch import nn
#from torch_pconv import PConv2d
import time
import numpy as np
import h5py
from scipy import signal
from torch.utils.data import Dataset

class n2nU_net(nn.Module):
    def __init__(self, in_chanel = 1, first_out_chanel=24, scaling_kernel_size=2, conv_kernel=3, batchNorm=False):
        """
        Args:
            in_chanel: Input chanels, default = 1
            first_out_chanel: First Output Chanel Size, default = 24
            scaling_kernel_size: kernel_size and stride for Maxpool2d and upsampling, defaullt = 2
            conv_kernel: kernel_size for Conv2d, default = 3
            batchhNorm: activate batchnorm after Conv-Layers, default = False
        """
        super(n2nU_net, self).__init__()
        self.norm= batchNorm
        self.encoder1 = nn.Conv2d(in_chanel, first_out_chanel, kernel_size=conv_kernel, padding="same")
        self.normLayer1 = nn.BatchNorm2d(first_out_chanel)
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=scaling_kernel_size, stride=scaling_kernel_size),
            nn.Conv2d(first_out_chanel, first_out_chanel, kernel_size=conv_kernel, padding="same")
        )
        self.normLayer2 = nn.BatchNorm2d(first_out_chanel)
        self.up = nn.ConvTranspose2d(first_out_chanel, first_out_chanel, kernel_size=scaling_kernel_size, stride=scaling_kernel_size)
        self.decoder1 = nn.Conv2d(first_out_chanel*2, first_out_chanel*2, kernel_size=conv_kernel, padding="same")
        self.normLayer3 = nn.BatchNorm2d(first_out_chanel*2)
        self.decoder2 = nn.Conv2d(first_out_chanel*2, first_out_chanel*2, kernel_size=conv_kernel, padding="same")
        self.normLayer4 = nn.BatchNorm2d(first_out_chanel*2)
        self.last_layer = nn.Conv2d(first_out_chanel*2, in_chanel, kernel_size=1, padding="same")
        #b, 1, 96, 128  input
        #b, 24, 96, 128 conv 3x3 lrelu(0.1) HIER
        #b, 24, 48, 64  maxpool 2x2
        #b, 24, 48, 64  conv 3x3 lrelu
        #b, 24, 96, 128 upsample 2x2
        #b, 48, 96, 128 cat mit HIER
        #b, 48, 96, 128 conv 3x3 lrelu
        #b, 48, 96, 128 conv 3x3 lrelu
        #b, 1, 96, 128  conv 1x1 linear activation
    def forward(self, x):
        skip = self.encoder1(x)
        x = nn.functional.leaky_relu(skip, 0.1)
        if self.norm:
            skip = self.normLayer1(skip)
        x = self.encoder2(x)
        x = nn.functional.leaky_relu(x, 0.1)    #b, 24, 48, 64  conv 3x3 lrelu
        if self.norm:
            x = self.normLayer2(x)
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)         #b, 48, 96, 128 cat mit HIER
        x = self.decoder1(x)
        x = nn.functional.leaky_relu(x, 0.1)    #b, 48, 96, 128 conv 3x3 lrelu
        if self.norm:
            x = self.normLayer3(x)
        x = self.decoder2(x)
        x = nn.functional.leaky_relu(x, 0.1)    #b, 48, 96, 128 conv 3x3 lrelu
        if self.norm:
            x = self.normLayer4(x)
        x = self.last_layer(x)                  #b, 1, 96, 128  conv 1x1
        return x

class U_Net(nn.Module):
    def __init__(self, in_chanel = 1, first_out_chanel=64, scaling_kernel_size=2, conv_kernel=3, batchNorm=False, skipLast=True, n2self_architecture=False):
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
        self.n2self_architecture = n2self_architecture
        self.skipLast = skipLast
        self.scaling_kernel_size = scaling_kernel_size
        self.encoder1 = doubleConv(in_chanel, first_out_chanel, conv_kernel=conv_kernel, norm=batchNorm, n2self_architecture=n2self_architecture)#todo: warum 3
        """ old version
        self.encoder2 = nn.Sequential(
            #nn.MaxPool2d(kernel_size=scaling_kernel_size, stride=scaling_kernel_size),
            BlurPool(in_chanel=first_out_chanel, kernel_size=scaling_kernel_size),  # Replace MaxPool2d with BlurPool
            doubleConv(first_out_chanel, first_out_chanel*2, conv_kernel, batchNorm),
        )
        self.encoder3 = nn.Sequential(
            #nn.MaxPool2d(kernel_size=scaling_kernel_size, stride=scaling_kernel_size),
            BlurPool(in_chanel=first_out_chanel*2, kernel_size=scaling_kernel_size),  # Replace MaxPool2d with BlurPool
            doubleConv(first_out_chanel*2, first_out_chanel*4, conv_kernel, batchNorm),
        )
        self.encoder4 = nn.Sequential(
            #nn.MaxPool2d(kernel_size=scaling_kernel_size, stride=scaling_kernel_size),
            BlurPool(in_chanel=first_out_chanel*4, kernel_size=scaling_kernel_size),  # Replace MaxPool2d with BlurPool
            doubleConv(first_out_chanel*4, first_out_chanel*8, conv_kernel, batchNorm),
        )
        self.encoder5 = nn.Sequential(
            #nn.MaxPool2d(kernel_size=scaling_kernel_size, stride=scaling_kernel_size),
            BlurPool(in_chanel=first_out_chanel*8, kernel_size=scaling_kernel_size),  # Replace MaxPool2d with BlurPool
            doubleConv(first_out_chanel*8, first_out_chanel*16, conv_kernel, batchNorm),
        )
        """
        self.encoder2 = down(first_out_chanel, scaling_kernel_size, conv_kernel, batchNorm, n2self_architecture)
        self.encoder3 = down(first_out_chanel*2, scaling_kernel_size, conv_kernel, batchNorm, n2self_architecture)
        self.encoder4 = down(first_out_chanel*4, scaling_kernel_size, conv_kernel, batchNorm, n2self_architecture)
        self.encoder5 = down(first_out_chanel*8, scaling_kernel_size, conv_kernel, batchNorm, n2self_architecture)

        self.decoder1 = Up(first_out_chanel*16, first_out_chanel*8, scaling_kernel_size, conv_kernel, batchNorm, skipConnection=self.skipLast, n2self_architecture=n2self_architecture)
        self.decoder2 = Up(first_out_chanel*8, first_out_chanel*4, scaling_kernel_size, conv_kernel, batchNorm, n2self_architecture=n2self_architecture)
        self.decoder3 = Up(first_out_chanel*4, first_out_chanel*2, scaling_kernel_size, conv_kernel, batchNorm, n2self_architecture=n2self_architecture)
        self.decoder4 = Up(first_out_chanel*2,first_out_chanel, scaling_kernel_size, conv_kernel, batchNorm, n2self_architecture=n2self_architecture)
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

class down(nn.Module):
    def __init__(self, in_chanel, scaling_kernel_size, conv_kernel_size, norm, n2self_architecture):
        super(down, self).__init__()
        self.n2self_architecture = n2self_architecture
        self.maxpool = nn.MaxPool2d(kernel_size=scaling_kernel_size, stride=scaling_kernel_size)
        self.blur = BlurPool(in_chanel=in_chanel, kernel_size=scaling_kernel_size)  # Replace MaxPool2d with BlurPool
        self.conv = doubleConv(in_chanel, in_chanel*2, conv_kernel_size, norm, n2self_architecture)

    def forward(self, x):
        if self.n2self_architecture:
            x = self.blur(x)
        else:
            x = self.maxpool(x)
        return self.conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channel, out_channel, scaling_kernel_size, conv_kernel, batchNorm, skipConnection=True, n2self_architecture=False):
        super().__init__()
        self.scaling_kernel_size = scaling_kernel_size
        self.n2self_architecture = n2self_architecture
        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=scaling_kernel_size, stride=scaling_kernel_size)
        self.reduceChanel = nn.Conv2d(in_channel, in_channel//2, kernel_size=1)
        if skipConnection == True:
            self.conv = doubleConv(in_channel, out_channel, conv_kernel, batchNorm, n2self_architecture)
        else: 
            self.conv = doubleConv(in_channel//2, out_channel, conv_kernel, batchNorm, n2self_architecture)
        self.apply(layer_init)

    def forward(self, x1, x2=None):
        if self.n2self_architecture:
            x = nn.functional.interpolate(x1, scale_factor=self.scaling_kernel_size, mode='bilinear', align_corners=True)
            x = self.reduceChanel(x)
        else:
            x = self.up(x1)
        if x2 is not None:  # skip konnection (immer auser bei N2Same)
            x = torch.cat((x, x2), dim=1)
        return self.conv(x)
    
class doubleConv(nn.Module):
    """
    conv(3x3) -> Batchnorm? -> Relu -> conv(3x3) - Batchnorm? -> Relu
    """
    def __init__(self, in_channels, out_channels, conv_kernel, norm=False, n2self_architecture=False):
        super().__init__()
        self.n2self_architecture = n2self_architecture
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
        if self.n2self_architecture:
            x = self.conv2(nn.functional.silu(x))
        else:
            x = self.conv2(nn.functional.relu(x))
        if self.norm:
            x = self.normLayer2(x)
        if self.n2self_architecture:
            return nn.functional.silu(x)
        else:
            return nn.functional.relu(x)

class BlurPool(nn.Module):
    def __init__(self, in_chanel, kernel_size=(1, 4)):
        super(BlurPool, self).__init__()
        self.kernel_size = kernel_size
        self.in_chanel = in_chanel

        # Erstellen eines 1D-Blur-Kernels
        a = torch.tensor([1., 3., 3., 1.], dtype=torch.float32)
        a = a / a.sum()
        
        # Wiederhole den Kernel für alle Kanäle
        a = a.view(1, 1, *kernel_size).repeat(in_chanel, 1, 1, 1)
        self.register_buffer('blur_kernel', a)
    
    def forward(self, x):
        # MaxPooling
        x = nn.functional.max_pool2d(x, kernel_size=self.kernel_size, stride=(1, 1), padding=0)
        padding_h = (self.kernel_size[0] - 1) // 2 #0
        padding_w = (self.kernel_size[1] - 1) // 2 #1
        padding_w = 2
        # Blur-Filter anwenden
        x = nn.functional.conv2d(x, self.blur_kernel, stride=self.kernel_size, padding=(padding_h, padding_w), groups=self.in_chanel)
        
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
        
        if isinstance(self.log_SNR, tuple):
            snr_sample = np.random.uniform(*self.log_SNR)
            snr = 10 ** snr_sample  # log10-uniform distribution
        else:
            snr_sample = self.log_SNR
            snr = 10 ** (self.log_SNR/10)

        #"""
        amp = 2 * np.sqrt(snr) / torch.abs(eq_das + 1e-10).max() #rescale so, max amplitude = 2*wrt(snr)
        eq_das *= amp

        # 1-10 Hz filtered Gaussian white noise
        gutter = 100
        noise = np.random.randn(self.nx, self.nt + 2*gutter)
        noise = torch.from_numpy(bandpass(noise, 1.0, 10.0, self.fs, gutter).copy())
        
        noise2 = np.random.randn(self.nx, self.nt + 2*gutter)
        noise2 = torch.from_numpy(bandpass(noise2, 1.0, 10.0, self.fs, gutter).copy())

        #print(f"snr-sample: {snr_sample}, snr: {snr}, noise.std: {noise.std()}, alpha: {alpha}")

        sample = eq_das + noise
        sample2 = eq_das + noise2
        scale2 = sample2.std(dim=-1, keepdim=True)
        scale = sample.std(dim=-1, keepdim=True)
        sample /= scale
        sample2 /= scale2
        #save_das_graph(eq_das.unsqueeze(0).unsqueeze(0), sample.unsqueeze(0).unsqueeze(0), scale.unsqueeze(0).unsqueeze(0))
        #           nois_image,         clean               noise         std der Daten (b,c,nx,1)  Ampllitude vom DAS (b)
        return sample.unsqueeze(0), eq_das.unsqueeze(0), noise.unsqueeze(0), scale.unsqueeze(0), amp, sample2.unsqueeze(0), noise2.unsqueeze(0), scale2.unsqueeze(0)
        #"""
        #return eq_das.unsqueeze(0)
"""
def save_das_graph(original, denoised, scale):
    def plot_das(data, title, ax, batch_idx):
        if isinstance(data, torch.Tensor):
            data = data.to('cpu').detach().numpy()
        data = data[batch_idx]
        for i in range(data.shape[1]):
            #std = data[0][i].std()
            #if std == 0:
                #std = 0.000000001
            sr = data[0][i]# / std
            #sr = data[0][i]
            ax.plot(sr + 3*i, c="k", lw=0.5, alpha=1)
        ax.set_title(title)
        ax.set_axis_off()
    # Erstelle eine Figur mit 3 Subplots
    fig, axs = plt.subplots(3, 1, figsize=(7, 10))
    # Erste Spalte - Batch 0
    plot_das(original, 'Clean', axs[0], 0)
    plot_das(denoised, 'sample', axs[1], 0)
    plot_das(denoised*scale, 'sample*scale', axs[2], 0)
    plt.tight_layout()
    plt.show()
"""
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
    current_path2 = Path(os.path.join(current_path, "das-solo"))
    store_path2 = Path(os.path.join(store_path, "das-solo"))
    store_path2.mkdir(parents=True, exist_ok=True)
    for file in os.listdir(current_path2):
        if file.endswith(".py"):
            print("copy: ", file)
            shutil.copyfile(os.path.join(current_path2, file), os.path.join(store_path2, file))
    print(f"python files are stored. Path: {store_path}")
    return store_path