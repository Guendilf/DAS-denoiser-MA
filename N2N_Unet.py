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

        #self.apply(layer_init)

    
    def forward(self, x):
        conv1 = self.conv1(x)
        output = self.net1(conv1)
        output = torch.cat((output, conv1), dim=2)
        output = self.net2(output)
        return output



    #def layer_init(layer, std=0.1, bias_const=0.0):
    #    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #        torch.nn.init.kaiming_normal_(layer.weight)
    #        torch.nn.init.constant_(layer.bias, 0)


class N2N_Unet_Claba(nn.Module):
    def __init__(self):
        super(N2N_Unet_Claba, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, padding='same')

        self.net1 = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, padding='same'),
            nn.Upsample(scale_factor=2, mode='nearest'),           
        )

        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=48, out_channels=3, kernel_size=1, padding='same')
            #linear activation
        )

        #self.apply(layer_init)

    
    def forward(self, x):
        conv1 = self.conv1(x)
        output = self.net1(conv1)
        output = torch.cat((output, conv1), dim=1)
        output = self.net2(output)
        return output
    

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



        #self.apply(layer_init)

    
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