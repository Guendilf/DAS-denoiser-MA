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
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

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
        skip1 = self.encoder1(x)  # (N, 4, 11, 2048)        real: (N, 4, 1482, 7500)
        skip2 = self.encoder2(skip1)  # (N, 8, 11, 512)     real: (N, 8, 1482, 1875)
        skip3 = self.encoder3(skip2)  # (N, 16, 11, 128)    real: (N, 16, 1482, 468)
        skip4 = self.encoder4(skip3)  # (N, 32, 11, 32)     real: (N, 32, 1482, 117)
        result = self.encoder5(skip4)  # (N, 64, 11, 8)     real: (N, 64, 1482, 29)
        #print(x.shape)
        #print(skip1.shape)
        #print(skip2.shape) 
        #print(skip3.shape) 
        #print(skip4.shape)
        #print(result.shape)
        #print()
        # Decoder with Skip Connections
        if self.skipLast:
            result = self.decoder1(result, skip4) # (N, 512, 16, 16  
        else:
            result = self.decoder1(result) # (N, 512, 16, 16)
        #print(result.shape)
        result = self.decoder2(result, skip3)  # (N, 256, 32, 32)
        #print(result.shape)
        result = self.decoder3(result, skip2)  # (N, 128, 64, 64)
        #print(result.shape)
        result = self.decoder4(result, skip1)  # (N, 64, 128, 128)
        #print(result.shape)
        
        result = self.final_conv(result)  # (N, 3, 128, 128)
        #print(result.shape)
        return result

class down(nn.Module):
    def __init__(self, in_chanel, scaling_kernel_size, conv_kernel_size, norm, n2self_architecture):
        super(down, self).__init__()
        self.n2self_architecture = n2self_architecture
        self.maxpool = nn.MaxPool2d(kernel_size=scaling_kernel_size, stride=scaling_kernel_size)
        if self.n2self_architecture:
            self.blur = BlurPool(in_chanel=in_chanel, kernel_size=scaling_kernel_size)  # Replace MaxPool2d with BlurPool
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=scaling_kernel_size, stride=scaling_kernel_size)
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

def orthogonal_init(module):
    if isinstance(module, (nn.Conv2d,)):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
class Sebastian_N2SUNet(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_dim=4):
        super().__init__()

        self.inconv = Sebastian_ConvBlock2(in_channels, hidden_dim, hidden_dim)

        self.down1 = Sebastian_Down2(hidden_dim, 2*hidden_dim)
        self.down2 = Sebastian_Down2(2*hidden_dim, 4*hidden_dim)
        self.down3 = Sebastian_Down2(4*hidden_dim, 8*hidden_dim)
        self.down4 = Sebastian_Down2(8*hidden_dim, 16*hidden_dim)

        self.up1 = Sebastian_Up2(16*hidden_dim, 8*hidden_dim)
        self.up2 = Sebastian_Up2(8*hidden_dim, 4*hidden_dim)
        self.up3 = Sebastian_Up2(4*hidden_dim, 2*hidden_dim)
        self.up4 = Sebastian_Up2(2*hidden_dim, hidden_dim)

        self.outconv = nn.Conv2d(hidden_dim, out_channels, kernel_size=(3, 5), padding='same') #(1, 2) #(3, 5)

        self.apply(orthogonal_init)

    def forward(self, x):    

        x0 = self.inconv(x)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        x = self.outconv(x)

        return x
class Sebastian_Down2(nn.Module):
        
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.pool = nn.MaxPool2d((1, 4), stride=(1, 1))
            w = torch.tensor([1., 3., 3., 1.]) / 8.
            self.w = nn.Parameter(w[None, None, None, :].repeat((in_channels, 1, 1, 1)), requires_grad=False)
            self.conv = Sebastian_ConvBlock2(in_channels, out_channels, out_channels)
    
        def forward(self, x):
            x = self.pool(x)
            x = nn.functional.conv2d(x, weight=self.w, stride=(1, 4), padding=(0, 2), groups=x.shape[1])
            x = self.conv(x)
            return x
class Sebastian_Up2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(1, 4), mode='bilinear')
        self.conv = Sebastian_ConvBlock2(in_channels+out_channels, out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)
class Sebastian_ConvBlock2(nn.Module):
    
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(3, 5), padding='same'),#(1, 2) #(3, 5)
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=(3, 5), padding='same'),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.layers(x)

class SyntheticNoiseDAS(Dataset):
    def __init__(self, eq_strain_rates, 
                 nx=11, nt=2048, eq_slowness=(1e-4, 5e-3), log_SNR=(-2,4),
                 gauge=4, fs=50.0, size=1000):
        self.eq_strain_rates = eq_strain_rates / eq_strain_rates.std(dim=-1, keepdim=True)
        self.nx = nx
        self.nt = nt
        self.eq_slowness = eq_slowness
        self.log_SNR = log_SNR
        self.gauge = gauge
        self.fs = fs
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        
        eq_strain_rate = self.eq_strain_rates[np.random.randint(0,len(self.eq_strain_rates))].clone()
        if np.random.random() < 0.5:
            eq_strain_rate = torch.flip(eq_strain_rate, dims=(0,))
        if np.random.random() < 0.5:
            eq_strain_rate *= -1
            
        slowness = np.random.uniform(*self.eq_slowness)
        if np.random.random() < 0.5:
            slowness *= -1
        eq_das = generate_synthetic_das(eq_strain_rate, self.gauge, self.fs, slowness, nx=self.nx)
        j = np.random.randint(0, eq_strain_rate.shape[-1]-self.nt+1)
        eq_das = eq_das[:,j:j+self.nt]

        if isinstance(self.log_SNR, tuple):
            snr_sample = np.random.uniform(*self.log_SNR)
        else:
            snr_sample = self.log_SNR
        snr = 10 ** snr_sample
        
        amp = 2 * np.sqrt(snr) / torch.abs(eq_das + 1e-10).max()
        eq_das *= amp

        gutter = 100
        noise = np.random.randn(self.nx, self.nt + 2*gutter)
        noise = torch.from_numpy(bandpass(noise, 1.0, 10.0, self.fs, gutter).copy())
        noise2 = np.random.randn(self.nx, self.nt + 2*gutter)
        noise2 = torch.from_numpy(bandpass(noise2, 1.0, 10.0, self.fs, gutter).copy())

        sample = eq_das + noise
        sample2 = eq_das + noise2
        scale = sample.std(dim=-1, keepdim=True)
        scale2 = sample2.std(dim=-1, keepdim=True)
        sample /= scale
        sample2 /= scale2
                
        return sample.unsqueeze(0), eq_das.unsqueeze(0), noise.unsqueeze(0), scale.unsqueeze(0), amp, sample2.unsqueeze(0), noise2.unsqueeze(0), scale2.unsqueeze(0)


class RealDAS(Dataset):
    def __init__(self, data, nx=128, nt=512, size=1000):
        
        self.data = torch.from_numpy(data.copy())
        self.nx, self.nt = nx, nt
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        
        n, nx_total, nt_total = self.data.shape
        nx = np.random.randint(0, nx_total - self.nx)
        nt = np.random.randint(0, nt_total - self.nt)
        
        patch = self.data[idx % n, nx:nx+self.nx, nt:nt+self.nt].clone()
        
        if np.random.random() < 0.5:
            patch = torch.flip(patch, dims=(0,))
        if np.random.random() < 0.5:
            patch = torch.flip(patch, dims=(1,))
        if np.random.random() < 0.5:
            patch *= -1 
        #       noise_images,        clean,                 noise,         std=1, amp=0, _, _, _
        return patch.unsqueeze(0), patch.unsqueeze(0), patch.unsqueeze(0), torch.ones((1, 11, 1)), 0, 0, 0, 0


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
def gerate_spezific_das(strain_rate,  nx=11, nt=2048, eq_slowness=0,
                 gauge=4, fs=50.0, station=None, start=None):
    if eq_slowness == 0:
        eq_slowness = 1/(gauge*fs)
    if station is not None:
        sample = strain_rate[station]
    else:
        sample = strain_rate
    if start is not None:
        sample = sample[start:start+nt]
    sample = generate_synthetic_das(sample, gauge, fs, eq_slowness, nx)
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


#from Mask.py
def mask_random(img, maskamount, mask_size=(4,4)):
    """
    TODO: is not saturated sampling (but in if section for mask_size=(1,1)) -> the same Pixel could be chosen multiple times -> in one picture there is less then required amount of masking
    Args
        img (tensor): Noisy images in form of (b,c,w,h) only for shape extraction
        maskamount (number): float for percentage masking; int for area amount masking
        mask_size (tupel): area that will be masked (w,h)
    Return
        mask (tensor): masked pixel are set to 1 (b,c,w,h)
        masked pixel (int): pixel that should be masked
    """
    total_area = img.shape[-1] * img.shape[-2]
    #if amount of pixel shoulld be masked
    if isinstance(maskamount, int):
        mask_percentage = maskamount*1/total_area
    else:
        mask_percentage = maskamount
        maskamount = int(np.round(mask_percentage*total_area/1))
        if maskamount == 0:
            maskamount = 1
    mask_area = mask_size[0] * mask_size[1]
    num_regions = int(np.round((mask_percentage * total_area) / mask_area))
    if num_regions == 0:
        num_regions = 1
    masks = []
    #fast methode for pixel only "select_random_pixel" or even with nn.functional.dropout
    #saturated sampling ensured through "torch.randperm" in select_random_pixels
    if mask_size == (1,1):
        for _ in range(img.shape[0]):
            mask = select_random_pixels((img.shape[1], img.shape[2],img.shape[3]), maskamount)
            masks.append(mask)
        mask = torch.stack(masks, dim=0)
        return mask, torch.count_nonzero(mask)
    
    else:
        print("maske gefährlich, weil nicht gecheckt")
        for _ in range(img.shape[0]):
            mask = torch.zeros(img.shape[-1], img.shape[-2], dtype=torch.float32)
            for _ in range(num_regions):        # generiere eine maske
                x = torch.randint(0, img.shape[-1] - mask_size[1] + 1, (1,))
                y = torch.randint(0, img.shape[-2] - mask_size[0] + 1, (1,))
                mask[x:x+mask_size[0], y:y+mask_size[1]] = 1
            masks.append(mask)
    
    mask = torch.stack(masks, dim=0)
    mask = mask.unsqueeze(1)  # (b, 1, w, h)
    mask = mask.expand(-1, img.shape[1], -1, -1) # (b, 3, w, h)
    return mask, torch.count_nonzero(mask)

def select_random_pixels(image_shape, num_masked_pixels):
    num_pixels = image_shape[0] * image_shape[1] * image_shape[2]
    # Erzeuge zufällige Indizes für die ausgewählten maskierten Pixel
    masked_indices = torch.randperm(num_pixels)[:num_masked_pixels]
    mask = torch.zeros(image_shape[0], image_shape[1], image_shape[2])
    # Pixel in Maske auf 1 setzen
    mask.view(-1)[masked_indices] = 1
    # Mache für alle Chanels
    #mask = mask.unsqueeze(0)
    return mask


def tv_norm(x):
    """
    Args:
        x: torch tensor to calculate its total variation
    Return:
        avarge total variation (avarge tv for eevery pixel)
    """
    h_diff = x[:, :, 1:, :] - x[:, :, :-1, :]
    w_diff = x[:, :, :, 1:] - x[:, :, :, :-1]

    h_diff_abs = torch.abs(h_diff)
    w_diff_abs = torch.abs(w_diff)

    tv_norm = torch.sum(h_diff_abs) + torch.sum(w_diff_abs)
    #durchschnitt bilden
    return tv_norm / (np.prod(x.shape))
