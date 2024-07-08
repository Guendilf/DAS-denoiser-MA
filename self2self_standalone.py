import numpy as np
import torch
import torch.nn.functional as F
#import network.Punet
import skimage.metrics
from argparse import ArgumentParser

#import network.myNet
import cv2
import os
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from torch import nn
from network.modify import PConv2d



from tensor_type import Tensor4d, Tensor
import math
from typing import Tuple, Union
import torch
from torch import nn

TupleInt = Union[int, Tuple[int, int]]


class PConv2d(nn.Module):
    ###############################################################################
    # BSD 3-Clause License
    #
    # Copyright (c) 2021, DesignStripe. All rights reserved.
    #
    # Author & Contact: Samuel Prevost (samuel@designstripe.com)
    # https://github.com/yangpuPKU/Self2Self_pytorch_implementation/blob/main/network/pconv.py
    #
    # original: "from torch_pconv import PConv2d"
    # changes: change mask from being (b,h,w) to be (b,c,h,w)
    ###############################################################################
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: TupleInt = 1,
        stride: TupleInt = 1,
        padding: TupleInt = 0,
        dilation: TupleInt = 1,
        bias: bool = False,
        legacy_behaviour: bool = False,
    ):

        super().__init__()
        self.legacy_behaviour = legacy_behaviour

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._to_int_tuple(kernel_size)
        self.stride = self._to_int_tuple(stride)
        self.padding = self._to_int_tuple(padding)
        self.dilation = self._to_int_tuple(dilation)
        self.use_bias = bias

        conv_kwargs = dict(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
            bias=False,
        )

        self.regular_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, **conv_kwargs)
        self.mask_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, **conv_kwargs)
        # Inits
        self.regular_conv.apply(
            lambda m: nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        )
        # the mask convolution should be a constant operation
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        for param in self.mask_conv.parameters():
            param.requires_grad = False

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(1, self.out_channels, 1, 1))
        else:
            self.register_parameter("bias", None)

        with torch.no_grad():
            # This is how nn._ConvNd initialises its weights
            nn.init.kaiming_uniform_(self.regular_conv.weight, a=math.sqrt(5))

            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.regular_conv.weight
                )
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias.view(self.out_channels), -bound, bound)

    def forward(self, x: Tensor4d, mask: Tensor4d) -> Tuple[Tensor4d, Tensor4d]:
        
        Tensor4d.check(x)
        batch, channels, h, w = x.shape
        Tensor[batch, channels, h, w].check(mask)

        if mask.dtype != torch.float32:
            raise TypeError(
                "mask should have dtype=torch.float32 with values being either 0.0 or 1.0"
            )

        if x.dtype != torch.float32:
            raise TypeError("x should have dtype=torch.float32")

        output = self.regular_conv(x * mask)
        _, _, conv_h, conv_w = output.shape

        update_mask: Tensor[batch, channels, conv_h, conv_w]
        mask_ratio: Tensor[batch, channels, conv_h, conv_w]
        with torch.no_grad():
            mask_ratio, update_mask = self.compute_masks(mask)

        if self.use_bias:
            if self.legacy_behaviour:
                
                output += self.bias
                output -= self.bias

            output *= mask_ratio  # Multiply by the sum(1)/sum(mask) ratios
            output += self.bias  # Add the bias *after* mask_ratio, not before !
            #output *= update_mask  # Nullify pixels outside the valid mask
        else:
            output *= mask_ratio

        return output, update_mask

    def compute_masks(self, mask: Tensor4d) -> Tuple[Tensor4d, Tensor4d]:
        
        update_mask = self.mask_conv(mask) #* self.in_channels
        
        mask_ratio = self.kernel_size[0] * self.kernel_size[1] / (update_mask + 1e-8)
        
        update_mask = torch.clamp(update_mask, 0, 1)
        mask_ratio *= update_mask
       
        return mask_ratio, update_mask

    @staticmethod
    def _to_int_tuple(v: TupleInt) -> Tuple[int, int]:
        if not isinstance(v, tuple):
            return v, v
        else:
            return v

    def set_weight(self, w):
        with torch.no_grad():
            self.regular_conv.weight.copy_(w)

        return self

    def set_bias(self, b):
        with torch.no_grad():
            self.bias.copy_(b.view(1, self.out_channels, 1, 1))

        return self

    def get_weight(self):
        return self.regular_conv.weight

    def get_bias(self):
        return self.bias


def layer_init(layer, std=0.1, bias_const=0.0):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        torch.nn.init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0) #Pconv has no bias
    elif isinstance(layer, PConv2d):
        torch.nn.init.kaiming_normal_(layer.regular_conv.weight)
        torch.nn.init.kaiming_normal_(layer.mask_conv.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0) #Pconv has no bias

class My_P_U_Net(nn.Module):
    def __init__(self, in_chanel = 3, batchNorm=False, dropout=0.3):
        """
        Args:
            in_chanel: Input chanels, default = 3
            batchhNorm: activate batchnorm in doubleConv-Layer, default = False
        """
        super(My_P_U_Net, self).__init__()
        
        self.inittial = PConv2d(in_chanel, 48)
        
        self.encoder1 = Encoder(48, 48, 0.1)
        self.encoder2 = Encoder(48, 48, 0.1)
        self.encoder3 = Encoder(48, 48, 0.1)
        self.encoder4 = Encoder(48, 48, 0.1)
        #pconv -> lerelu -> maxpool -> pconv -> lerelu -> decoder
        self.encoder5 = Encoder(48, 48, 0.1)
        self.encoder6 = Encoder(48, 48, 0.1, maxpool=False)


        self.decoder1 = Up(48, 48, 96, dropout)
        self.decoder2 = Up(96, 48, 96, dropout)
        self.decoder3 = Up(96, 48, 96, dropout)
        self.decoder4 = Up(96, 48, 96, dropout)

        self.up = nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2)

        self.last_decoder_block_droopout = nn.Dropout(dropout)
        self.last_decoder_block_1a = nn.Sequential(
            nn.Conv2d(96+in_chanel, 64, kernel_size=3, stride=1, padding='valid'),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )
        self.last_decoder_block_1b = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding='valid'),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout), #belonggs to "final_coonv" but because of extra padding itt was moved
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(32, in_chanel, kernel_size=3, stride=1, padding='valid'),
            nn.Sigmoid(),
            
        )
        self.apply(layer_init)

    def forward(self, x, mask):
        # Encoder
        init, mask = self.inittial(x, mask)     # (N, 48, 128, 128)
        skip1, mask = self.encoder1(init, mask)  # (N, 48, 64, 64)
        skip2, mask = self.encoder2(skip1, mask)  # (N, 48, 32, 32)
        skip3, mask = self.encoder3(skip2, mask)  # (N, 48, 16, 16)
        skip4, mask = self.encoder4(skip3, mask)  # (N, 48, 8, 8)
        result, mask = self.encoder5(skip4, mask)  # (N, 48, 4, 4)
        result, mask = self.encoder6(result, mask)  # (N, 48, 4, 4)   !macht kein Sinn

        # Decoder

        result = self.decoder1(result, skip4) # (N, 96, 8, 8)
        result = self.decoder2(result, skip3)  # (N, 96, 16, 16)
        result = self.decoder3(result, skip2)  # (N, 96, 32, 32)
        result = self.decoder4(result, skip1)  # (N, 96, 64, 64)

        padding_shape = (1, 1, 1, 1)
        result = self.up(result)  # (N, 96, 128, 128)
        result = torch.cat((result, x), dim=1)  # (N, 96+c, 128, 128)
        result = self.last_decoder_block_droopout(result)
        result = nn.functional.pad(result, padding_shape, mode="replicate") # (N, 96+c, 130, 130)
        result = self.last_decoder_block_1a(result)  # (N, 64, 128, 128)
        result = nn.functional.pad(result, padding_shape, mode="replicate") # (N, 96+c, 130, 130)
        result = self.last_decoder_block_1b(result)  # (N, 32, 128, 128)
        result = nn.functional.pad(result, padding_shape, mode="replicate") # (N, 32, 130, 130)

        result = self.final_conv(result)  # (N, c, 128, 128)
        
        return result
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channel, skip_chanel, out_channel, dropout, lr=0.1):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channel, in_channel, kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel+skip_chanel, out_channel, kernel_size=3, stride=1, padding='valid'),
            nn.LeakyReLU(lr),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding='valid'),
            nn.LeakyReLU(lr),
        )
        
        self.apply(layer_init)

    def forward(self, x1, x2=None):
        padding_shape = (1, 1, 1, 1)
        x = self.up(x1)
        x = self.dropout1(x)
        x = torch.cat((x, x2), dim=1)
        x = nn.functional.pad(x, padding_shape, mode="replicate")
        x = self.conv1(x)
        x = nn.functional.pad(x, padding_shape, mode="replicate")
        x = self.conv2(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel, lr=0.1, maxpool=True):
        super().__init__()

        self.maxpool = maxpool
        self.lr = lr
        self.pconv = PConv2d(in_channel, out_channel, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, mask):
        padding_shape = (1,1,1,1)
        x = nn.functional.pad(x, padding_shape, "replicate")
        mask = nn.functional.pad(mask, padding_shape, "constant", value=1)
        x, mask = self.pconv(x,mask)
        x = nn.LeakyReLU(self.lr)(x)
        mask = nn.LeakyReLU(self.lr)(mask)
        if self.maxpool:
            x = self.pool(x)
            mask = self.pool(mask)
        return x, mask



def add_noise_snr(x, snr_db):
    """
    Args:
        input_tensor (torch.Tensor): Der Eingabetensor (b, c, w, h).
        snr_db (float): Das gewünschte Signal-Rausch-Verhältnis in Dezibel (dB).
    Return:
        noise image: tensor of x + noise
        alpha: skaling faktor for noise - also interpreted as sigma
    """
    noise = torch.randn_like(x)
    snr_linear = 10 ** (snr_db / 10.0)
    snr_linear = snr_db
    Es = torch.sum(x**2)
    En = torch.sum(noise**2)
    alpha = torch.sqrt(Es/(snr_linear*En))
    noise = x + noise * alpha
    return noise, alpha.item()
def show_tensor_as_picture(img):
    img = img.cpu().detach()
    if len(img.shape)==4:
        img = img[0]
    if img.shape[0] == 3 or img.shape[0] == 1:
        img = img.permute(1,2,0)
    if img.shape[0] == 1:
        plt.imshow(img, cmap='gray', interpolation='nearest')
    else:
        plt.imshow(img, interpolation='nearest')
    plt.show()
def calculate_psnr(x,y):
    max_intensity = 1.0  # weil Vild [0, 1]
    if torch.is_tensor(x):
        if torch.is_tensor(y):
            mse = torch.mean((x-y)**2)
            psnr = 10 * torch.log10((max_intensity ** 2) / mse)
    else:
        mse = np.mean((x-y)**2)
        psnr = 10 * np.log10((max_intensity ** 2) / mse)
    return psnr

def data_arg(x, is_flip_lr, is_flip_ud):
    if is_flip_lr > 0:
        x = torch.flip(x, dims=[2])
    if is_flip_ud > 0:
        x = torch.flip(x, dims=[3])
    return x
def load_np_image(path, is_scale=True):
    img = cv2.imread(path, -1)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    if is_scale:
        img = np.array(img).astype(np.float32) / 255.
    return img

def get_output(noisy, model, drop_rate=0.3, bs=1, device='cpu'):
    noisy_tensor = torch.tensor(noisy).permute(0,3,1,2).to(device)
    is_flip_lr = np.random.randint(2)
    is_flip_ud = np.random.randint(2)
    noisy_tensor = data_arg(noisy_tensor, is_flip_lr, is_flip_ud)
    # mask_tensor = torch.ones([bs, model.width, model.height]).to(device)
    mask_tensor = torch.ones(noisy_tensor.shape).to(device)
    mask_tensor = F.dropout(mask_tensor, drop_rate) * (1-drop_rate)
    input_tensor = noisy_tensor * mask_tensor#.unsqueeze(1)
    output = model(input_tensor, mask_tensor) #(b,c,w,h)
    output = data_arg(output, is_flip_lr, is_flip_ud)
    #output_numpy = output.detach().cpu().numpy().transpose(0,2,3,1)
    output_numpy = output.permute(0,2,3,1)
    return output_numpy

def get_loss(noisy, model, drop_rate=0.3, bs=1, device='cpu'):
    #daten zwischen 0,1 # rauschen
    noisy_tensor = torch.tensor(noisy).permute(0,3,1,2).to(device)
    is_flip_lr = np.random.randint(2)
    is_flip_ud = np.random.randint(2)
    noisy_tensor = data_arg(noisy_tensor, is_flip_lr, is_flip_ud)
    # mask_tensor = torch.ones([bs, model.width, model.height]).to(device)
    mask_tensor = torch.ones(noisy_tensor.shape).to(device)
    mask_tensor = F.dropout(mask_tensor, drop_rate) * (1-drop_rate)
    input_tensor = noisy_tensor * mask_tensor#.unsqueeze(1)
    output = model(input_tensor, mask_tensor)
    observe_tensor = 1.0 - mask_tensor#.unsqueeze(1)
    loss = torch.sum((output-noisy_tensor).pow(2)*(observe_tensor)) / torch.count_nonzero(observe_tensor).float()
    return loss

def train(file_path, args, dataLoader, pathTMP=1, fileList=1):
    #original variante
    if file_path == True:
        print(file_path)    #./Self2Self_pytorch_implementation/testsets/Set9/5.png
        gt1 = load_np_image(pathTMP+fileList[1])
        gt2 = load_np_image(pathTMP+fileList[3])
        gt = np.concatenate((gt1, gt2))
        b , w, h, c = gt.shape
        model_path = file_path[0:file_path.rfind(".")] + "/" + str(args.sigma) + "/model_" + args.model_type + "/"
        os.makedirs(model_path, exist_ok=True)
        #noisy = util.add_gaussian_noise(gt, model_path, args.sigma, bs=args.bs)
        noisy, true_sigma = add_noise_snr(torch.tensor(gt), args.sigma)
        print('noisy shape:', noisy.shape)
        print('image shape:', gt.shape)
    
        #model = network.Punet.Punet(channel=c, width=w, height=h, drop_rate=args.drop_rate).to(args.device)
        model = My_P_U_Net(in_chanel = c, batchNorm=True, dropout=args.drop_rate).to(args.device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        avg_loss = 0
        for step in range(args.iteration):
            # one step
            loss = get_loss(noisy, model, drop_rate=args.drop_rate, bs=args.bs, device=args.device)
            avg_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.cuda.device(args.device):
                torch.cuda.empty_cache()
            
            # test
            if (step+1) % args.test_frequency == 0:
                # model.eval()
                print("After %d training step(s)" % (step + 1),
                    "loss  is {:.9f}".format(avg_loss / args.test_frequency))
                final_image = np.zeros(gt.shape)
                for j in range(args.num_prediction):
                    output_numpy = get_output(noisy, model, drop_rate=args.drop_rate, bs=args.bs, device=args.device)
                    final_image += output_numpy
                    with torch.cuda.device(args.device):
                        torch.cuda.empty_cache()
                final_image = np.squeeze(np.uint8(np.clip(final_image / args.num_prediction, 0, 1) * 255))
                cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '.png', final_image)
                PSNR = skimage.metrics.peak_signal_noise_ratio(gt[0], final_image.astype(np.float32)/255.0)
                print("psnr = ", PSNR)
                with open(args.log_pth, 'a') as f:
                    f.write("After %d training step(s), " % (step + 1))
                    f.write("loss  is {:.9f}, ".format(avg_loss / args.test_frequency)) 
                    f.write("psnr  is {:.4f}".format(PSNR))
                    f.write("\n")
                avg_loss = 0
                model.train()
    #with celebA
    else:
        c=3
        model = My_P_U_Net(in_chanel = c, batchNorm=True, dropout=args.drop_rate).to(args.device)
        model.train()
        
        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # begin training
        avg_loss = 0
        for step in tqdm(range(args.iteration)):
            for batch_idx, (gt, label) in enumerate((dataLoader)):
                gt = gt.permute(0,2,3,1).to(args.device)
                noisy, true_sigma = add_noise_snr(gt, args.sigma)
                # one step
                loss = get_loss(noisy, model, drop_rate=args.drop_rate, bs=args.bs, device=args.device)
                avg_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.cuda.device(args.device):
                    torch.cuda.empty_cache()
                
                # test
                if (step+1) % args.test_frequency == 0:
                    step = batch_idx
                    
                    final_image = torch.zeros(gt.shape).to(args.device) #(b,w,h,c)
                    with torch.no_grad():
                        for j in range(args.num_prediction):
                            output_numpy = get_output(noisy, model, drop_rate=args.drop_rate, bs=args.bs, device=args.device)
                            final_image += output_numpy
                            with torch.cuda.device(args.device):
                                torch.cuda.empty_cache()
                    final_image =(torch.clip(final_image / args.num_prediction, 0, 1) * 255) #(w,h,c) [0:255]
                    save_image = final_image.cpu().numpy()
                    save_image = np.uint8(save_image ) #(b,w,h,c) [0:255]

                    PSNR = calculate_psnr(gt, final_image/255).item()
                    print("After %d training step(s)" % (step + 1),
                        "loss  is {:.9f}".format(avg_loss / args.test_frequency))
                    print("psnr = ", PSNR)

                    avg_loss = 0
                    model.train()
    
    return PSNR
    
def main(args):
    planepicture = args.planepicture
    avg_psnr = 0
    count = 0
    if planepicture == True:
        """
        instructions for using the original implementation from https://github.com/yangpuPKU/Self2Self_pytorch_implementation/blob/main/main.py:
        This file is in the folder "Self2Self_pytorch_implementation"
        thhe original Pictture from https://github.com/scut-mingqinchen/Self2Self/tree/master/testsets/Set9 shhould be saved in "Self2Self_pytorch_implementation/testsets/Set9/" as "[something].png"
        make sure there exists the following follder: "Self2Self_pytorch_implementation/logs"
        """
        path = './Self2Self_pytorch_implementation/testsets/Set9/'
        path = args.path
        file_list = os.listdir(path)
        with open(args.log_pth, 'w') as f:
            f.write("Self2self algorithm!\n")
        for file_name in file_list:
            #if not os.path.isdir(path + file_name):
            if "." in file_name:
                PSNR = train(path+file_name, args, dataLoader, path, file_list)
                avg_psnr += PSNR
                count += 1
                break
        with open(args.log_pth, 'a') as f:
            f.write('average psnr is {:.4f}'.format(avg_psnr/count))
    else:
        celeba_dir = 'dataset/celeba_dataset'
        transform_noise = transforms.Compose([
                transforms.CenterCrop((128,128)),
                #transforms.Resize((512,512)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.float()),
                #transforms.Lambda(lambda x:  x * 255),
            ])
        dataset = datasets.CelebA(root=celeba_dir, split='train', transform=transform_noise, download=False)
        dataset = torch.utils.data.Subset(dataset, list(range(2)))
        dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)

        PSNR = train(planepicture, args, dataLoader)
        avg_psnr += PSNR
        count += 1
    

def build_args():
    parser = ArgumentParser()
    
    parser.add_argument("--iteration", type=int, default=450000)#1000
    parser.add_argument("--test_frequency", type=int, default=1000)
    parser.add_argument("--drop_rate", type=float, default=0.3)
    #original sigma use:
    #parser.add_argument("--sigma", type=float, default=127.0) #default was 25 -> sigma=0.09
    parser.add_argument("--sigma", type=float, default=2) # 2db = 0.5187782049179077
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--model_type", type=str, default='dropout')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_prediction", type=int, default=50)
    parser.add_argument("--log_pth", type=str, default='./Self2Self_pytorch_implementation/logs/log.txt')
    parser.add_argument("--path", type=str, default='./Self2Self_pytorch_implementation/testsets/Set9/')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--planepicture", type=str, default='False')
    
    args = parser.parse_args()
    return args
            
if __name__ == "__main__":
    args = build_args()
    main(args)
    #original Output:
    #After 1000 training step(s) loss  is 0.013080759
    #psnr =  26.373334640579714