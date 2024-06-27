import torch
from torch import nn

def layer_init(layer, std=0.1, bias_const=0.0):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        torch.nn.init.kaiming_normal_(layer.weight)
        torch.nn.init.constant_(layer.bias, 0)


class U_Net(nn.Module):
    def __init__(self, in_chanel = 3, firstConv = 64, maxpool_kernel = (1,4), conv_kernel=(3,5), batchNorm=False, dropout=-1):
        """
        Args:
            in_chanel: Input chanels, default = 3
            firstConv: Output chanels of first conv layer (ooften referd as init-layer), default =64
            maxpool_kernel: kernel size for maxpool and upsampling, default =(1,4) - no sampling in DAS chanel but only in time
            conv_kernel: kernel size for conv-layer, default =(3,5)
            batchhNorm: activate batchnorm in doubleConv-Layer, default = False
            dropout: Dropout rate, default -1 -> deaktivate
        """
        super(U_Net, self).__init__()
        
        self.dropout = dropout
        self.encoder1 = doubleConv(in_chanel, firstConv, conv_kernel, batchNorm)
        
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=maxpool_kernel, stride=2),
            doubleConv(firstConv, firstConv*2, conv_kernel, batchNorm),
        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=maxpool_kernel, stride=2),
            doubleConv(firstConv*2, firstConv*4, conv_kernel, batchNorm),
        )
        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=maxpool_kernel, stride=2),
            doubleConv(firstConv*4, firstConv*8, conv_kernel, batchNorm),
        )
        self.encoder5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=maxpool_kernel, stride=2),
            doubleConv(firstConv*8, firstConv*16, conv_kernel, batchNorm),
        )

        self.decoder1 = Up(firstConv*16, firstConv*8, batchNorm, maxpool_kernel, conv_kernel)
        self.decoder2 = Up(firstConv*8, firstConv*4, batchNorm, maxpool_kernel, conv_kernel)
        self.decoder3 = Up(firstConv*4, firstConv*2, batchNorm, maxpool_kernel, conv_kernel)
        self.decoder4 = Up(firstConv*2,firstConv, batchNorm, maxpool_kernel, conv_kernel)
        self.final_conv = nn.Conv2d(firstConv, in_chanel, kernel_size=1)
        self.apply(layer_init)

    def forward(self, x):
        # Encoder
        skip1 = self.encoder1(x)  # (N, 64, 128, 128)
        skip2 = self.encoder2(skip1)  # (N, 128, 64, 64)
        skip3 = self.encoder3(skip2)  # (N, 256, 32, 32)
        skip4 = self.encoder4(skip3)  # (N, 512, 16, 16)
        result = self.encoder5(skip4)  # (N, 1024, 8, 8)

        # Decoder with Skip Connections

        result = self.decoder1(result) # (N, 512, 16, 16)
        result = self.decoder2(result, skip3)  # (N, 256, 32, 32)
        result = self.decoder3(result, skip2)  # (N, 128, 64, 64)
        result = self.decoder4(result, skip1)  # (N, 64, 128, 128)
        
        result = self.final_conv(result)  # (N, 3, 128, 128)
        return result
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channel, out_channel, batchNorm, maxpool_kernel, conv_kernel):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=maxpool_kernel, stride=2)
        

        self.conv = doubleConv(in_channel//2, out_channel, conv_kernel, batchNorm)
        self.apply(layer_init)

    def forward(self, x1, x2=None):
        x = self.up(x1)
        if x2 is not None:  # skip konnection (immer auser bei N2Same)
            x = torch.cat((x, x2), dim=1)
        return self.conv(x)
    
class doubleConv(nn.Module):
    """
    conv(3x3) -> Batchnorm? -> Relu -> conv(3x3) - Batchnorm? -> Relu
    """
    def __init__(self, in_channels, out_channels, conv_kernel, norm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel, padding=1)
        self.normLayer1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=conv_kernel, padding=1)
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