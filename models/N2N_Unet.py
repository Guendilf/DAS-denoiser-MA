#from N2N paper - DAS-N2N: Machine learning Distributed Acoustic Sensing (DAS) signal denoising without clean data

#input (128,96,1)
#conv2D(128,96,24) 3x3
#leaky Relu
#maxpool 2x2
#conv2d(64,48,24) 3x3
#leaky Relu
#UpSampling2D 2x2
#concat mit 1. conv
#conv2D(128,96,48) 3x3
#leaky Relu
#conv2D(128,96,24) 3x3
#leaky Relu
#conv2D(128,96,1) 1x1

import torch
from torch import nn
from torch_pconv import PConv2d


#for use on CelebA - Paper of N2N - Quelle 2        richtig implementiert
class N2N_Unet_DAS(nn.Module):
    def __init__(self):
        super(N2N_Unet_DAS, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=3, padding='same')

        self.net1 = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, padding='same'),
            nn.Upsample(scale_factor=2, mode='nearest'),           
        )

        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=48, out_channels=1, kernel_size=1, padding='same')
            #linear activation
        )

        self.apply(layer_init)

    
    def forward(self, x):
        conv1 = self.conv1(x)
        output = self.net1(conv1)
        output = torch.cat((output, conv1), dim=2)
        output = self.net2(output)
        return output



def layer_init(layer, std=0.1, bias_const=0.0):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        torch.nn.init.kaiming_normal_(layer.weight)
        torch.nn.init.constant_(layer.bias, 0)


#for use on CelebA - Original Paper of N2N - Quelle 1
class N2N_Orig_Unet(nn.Module):
    def __init__(self, input_chanels, output_chanels):
        super(N2N_Orig_Unet, self).__init__()

        self.input_chanels = input_chanels
        self.output_chanels = output_chanels

        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_chanels, out_channels=48, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),         
        )

        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  
        )
        self.net3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  
        )
        self.net4 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  
        )
        self.net5 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='nearest'),  
        )



        self.net6 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='nearest'),  
        )
        self.net7 = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=96, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='nearest'),  
        )
        self.net8 = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=96, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='nearest'),  
        )
        self.net9 = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=96, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='nearest'),  
        )
        self.net10 = nn.Sequential(
            nn.Conv2d(in_channels=96+self.input_chanels, out_channels=64, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=32, out_channels=self.output_chanels, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
        )

        self.apply(layer_init)

    
    def forward(self, x):
        pool1 = self.net1(x)
        pool2 = self.net2(pool1)
        pool3 = self.net3(pool2)
        pool4 = self.net4(pool3)
        pool5 = self.net5(pool4)

        output = torch.cat((pool5, pool4), dim=1) #TODO: richtige Dim?
        output = self.net6(output)
        output = torch.cat((output, pool3), dim=1)
        output = self.net7(output)
        output = torch.cat((output, pool2), dim=1)
        output = self.net8(output)
        output = torch.cat((output, pool1), dim=1)
        output = self.net9(output)
        output = torch.cat((output, x), dim=1)
        output = self.net10(output)

        return output
    

class TestNet(nn.Module):
    def __init__(self, input_chanels, output_chanels):
        super(TestNet, self).__init__()

        self.input_chanels = input_chanels
        self.output_chanels = output_chanels

        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_chanels, out_channels=48, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=48, out_channels=3, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),

        )


        self.apply(layer_init)

    
    def forward(self, x):
        output = self.net1(x)
        return output





#original U-Net aritecckture - Quelle 30
class U_Net_origi(nn.Module):
    def __init__(self, mask=None):
        super(U_Net_origi, self).__init__()

        self.encoder1 = nn.Sequential(          #572x572
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),                          #568x568
        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  #284x284
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),                              #280x280
        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  #140x140
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),                              #136x136
        )
        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  #68x68
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),                              #64x64
        )
        self.encoder5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  #32x32
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3),
            nn.ReLU(),                              #28x28
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.decoder1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(), 
            nn.Conv2d(64, 3, kernel_size=1),
            nn.ReLU(), 
        )
        self.apply(layer_init)

    def forward(self, x):
        if not x.shape[2] == 572:
            print("WARNING: Shaps not testet for diffrent resolutions then 572x572")
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)
        result = self.encoder5(skip4)
        result = self.upsample(result)
                            #512,   1024
        result = torch.cat((skip4, result), 0) #TODO: check ob fesatures in dim 0 sind
        result = self.decoder1(result)
        result = torch.cat((skip3, result), 0)
        result = self.decoder2(result)
        result = torch.cat((skip2, result), 0)
        result = self.decoder3(result)
        result = torch.cat((skip1, result), 0)
        result = self.decoder4(result)
        return result

#angepasst an 128x128 bilder
class U_Net(nn.Module):
    def __init__(self, in_chanel = 3, batchNorm = False):
        super(U_Net, self).__init__()

        self.encoder1 = doubleConv(in_chanel, 64, batchNorm)
        
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            doubleConv(64, 128, batchNorm),
        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            doubleConv(128, 256, batchNorm),
        )
        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            doubleConv(256, 512, batchNorm),
        )
        self.encoder5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            doubleConv(512, 1024, batchNorm),
        )

        self.decoder1 = Up(1024, 512, batchNorm)
        self.decoder2 = Up(512, 256, batchNorm)
        self.decoder3 = Up(256, 128, batchNorm)
        self.decoder4 = Up(128,64, batchNorm)
        self.final_conv = nn.Conv2d(64, in_chanel, kernel_size=1)
        self.apply(layer_init)

    def forward(self, x):
        # Encoder
        skip1 = self.encoder1(x)  # (N, 64, 128, 128)
        skip2 = self.encoder2(skip1)  # (N, 128, 64, 64)
        skip3 = self.encoder3(skip2)  # (N, 256, 32, 32)
        skip4 = self.encoder4(skip3)  # (N, 512, 16, 16)
        result = self.encoder5(skip4)  # (N, 1024, 8, 8)

        # Decoder with Skip Connections
        result = self.decoder1(result, skip4) # (N, 512, 16, 16)
        result = self.decoder2(result, skip3)  # (N, 256, 32, 32)
        result = self.decoder3(result, skip2)  # (N, 128, 64, 64)
        result = self.decoder4(result, skip1)  # (N, 64, 128, 128)
        
        result = self.final_conv(result)  # (N, 3, 128, 128)
        return result
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_chanel, out_chanel, batchNorm):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_chanel, in_chanel // 2, kernel_size=2, stride=2)
        self.conv = doubleConv(in_chanel, out_chanel, batchNorm)
        self.apply(layer_init)

    def forward(self, x1, x2):
        x = self.up(x1)
        x = torch.cat((x, x2), dim=1)
        return self.conv(x)
    
class doubleConv(nn.Module):
    """
    conv(3x3) -> Batchnorm? -> Relu -> conv(3x3) - Batchnorm? -> Relu
    """
    def __init__(self, in_channels, out_channels, norm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.normLayer1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
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
        return nn.ReLU(x)




#for use on CelebA - Paper Cut2Self - Quelle 4
class Cut2Self(nn.Module):
    def __init__(self, mask=None):
        super(Cut2Self, self).__init__()

        self.mask = mask
        #kernelsize von 3 wird angenommen, da nicht spezifiziert ist
        self.input_layer = PConv2d(in_channels=3, out_channels=48, kernel_size=3, stride=1, padding=0)
        self.encoder_conv = PConv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=0)
        self.encoder = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.decoder1 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(144, 96, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(144, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )

        self.apply(layer_init)

    
    def forward(self, x, mask=None):
        #encoder Block
        if mask==None:
            mask=self.mask
        conv1, mask = self.input_layer(x, mask)
        conv2, mask = self.encoder_conv(conv1, mask)
        conv2 = self.encoder(conv2)
        mask = nn.MaxPool2d(kernel_size=2, stride=2)(mask)
        conv3, mask = self.encoder_conv(conv2, mask)
        conv3 = self.encoder(conv3)
        mask = nn.MaxPool2d(kernel_size=2, stride=2)(mask)
        conv4, mask = self.encoder_conv(conv3, mask)
        conv4 = self.encoder(conv4)
        mask = nn.MaxPool2d(kernel_size=2, stride=2)(mask)
        conv5, mask = self.encoder_conv(conv4, mask)
        conv5 = self.encoder(conv5)
        mask = nn.MaxPool2d(kernel_size=2, stride=2)(mask)
        conv6, mask = self.encoder_conv(conv5, mask)
        conv6 = self.encoder(conv6)

        #decoder Block
        output = self.upsample(conv6)
        output = torch.cat((output, conv5), dim=2)
        output = self.decoder1(output)

        output = self.upsample(conv5)
        output = torch.cat((output, conv4), dim=2)
        output = self.decoder2(output)

        output = self.upsample(conv4)
        output = torch.cat((output, conv3), dim=2)
        output = self.decoder2(output)

        output = self.upsample(conv3)
        output = torch.cat((output, conv2), dim=2)
        output = self.decoder2(output)

        output = self.upsample(conv2)
        output = torch.cat((output, conv1), dim=2)
        output = self.output_layer(output)
        return output