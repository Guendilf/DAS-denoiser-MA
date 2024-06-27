import torch
from torch import nn
from models.pconv_modified import PConv2d
### conv layer has diffrent padding then normal


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

class P_U_Net(nn.Module):
    def __init__(self, in_chanel = 3, batchNorm=False, dropout=0.3):
        """
        Args:
            in_chanel: Input chanels, default = 3
            batchhNorm: activate batchnorm in doubleConv-Layer, default = False
        """
        super(P_U_Net, self).__init__()
        
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
        skip1, mask = self.encoder1(init, mask)  # (N, 64, 64, 128)
        skip2, mask = self.encoder2(skip1, mask)  # (N, 128, 32, 64)
        skip3, mask = self.encoder3(skip2, mask)  # (N, 256, 16, 32)
        skip4, mask = self.encoder4(skip3, mask)  # (N, 512, 8, 16)
        result, mask = self.encoder5(skip4, mask)  # (N, 1024, 4, 4)
        result, mask = self.encoder6(result, mask)  # (N, 1024, 4, 4)   !macht kein Sinn

        # Decoder

        result = self.decoder1(result, skip4) # (N, 512, 16, 16)
        result = self.decoder2(result, skip3)  # (N, 256, 32, 32)
        result = self.decoder3(result, skip2)  # (N, 128, 64, 64)
        result = self.decoder4(result, skip1)  # (N, 64, 128, 128)

        padding_shape = (1, 1, 1, 1)
        result = self.up(result)
        result = torch.cat((result, x), dim=1)
        result = self.last_decoder_block_droopout(result)
        result = nn.functional.pad(result, padding_shape, mode="replicate")
        result = self.last_decoder_block_1a(result)
        result = nn.functional.pad(result, padding_shape, mode="replicate")
        result = self.last_decoder_block_1b(result)
        result = nn.functional.pad(result, padding_shape, mode="replicate")

        result = self.final_conv(result)  # (N, 64, 128, 128)
        
        return result, mask
    
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
        self.pconv = PConv2d(in_channel, out_channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, mask):
        x, mask = self.pconv(x,mask)
        x = nn.LeakyReLU(self.lr)(x)
        mask = nn.LeakyReLU(self.lr)(mask)
        if self.maxpool:
            x = self.pool(x)
            mask = self.pool(mask)
        return x, mask
