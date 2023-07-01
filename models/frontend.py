import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import numpy as np

class SincConv(nn.Module):
    """_summary_
    This code is from https://github.com/clovaai/aasist/blob/main/models/RawNet2Spoof.py
    """
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(
        self,
        #device,
        out_channels,
        kernel_size,
        in_channels=1,
        sample_rate=16000,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
    ):
        super().__init__()

        if in_channels != 1:

            msg = (
                "SincConv only support one input channel (here, in_channels = {%i})"
                % (in_channels))
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        #self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        # initialize filterbanks using Mel scale
        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)  # Hz to mel conversion
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)  # Mel to Hz conversion
        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                  (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)

    def forward(self, x):
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(
                2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * np.sinc(
                2 * fmin * self.hsupp / self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = torch.Tensor(np.hamming(
                self.kernel_size)) * torch.Tensor(hideal)

        band_pass_filter = self.band_pass.to(x.device)

        self.filters = (band_pass_filter).view(self.out_channels, 1,
                                               self.kernel_size)

        return F.conv1d(
            x,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )

class Conv2DBlockS(nn.Module):
    """__summary__
    This block implements squeeze and excitation operation on ResNet block of AASIST.
    for Rawformer-S
    (https://github.com/clovaai/aasist/blob/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/models/AASIST.py#L413)
    """
    
    def __init__(self, in_channels: int, out_channels: int, se_reduction: int=8, is_first_block: bool=False):
        """_summary_

        Args:
            in_channels (int): num of input channels
            out_channels (int): num of output channels
            se_reduction (int, optional): reduction factor for squeeze and excitation of channels. Defaults to 8.
            is_first_block (bool, optional): if this is the first block must be True. Defaults to False.
        """
        
        super(Conv2DBlockS, self).__init__()
        
        self.preprocess = nn.Sequential()
        if not is_first_block:
            self.preprocess = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.SELU(inplace=True)
            )
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 3), padding=(1, 1), stride=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.SELU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(2, 3), padding=(0, 1), stride=1),
        )        
        
        self.downsampler = nn.Sequential()
        if in_channels != out_channels:
            self.downsampler = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding=(0, 1), kernel_size=(1, 3), stride=1)
            )
        
        self.pooling = nn.MaxPool2d(kernel_size=(1, 3))
        
    def forward(self, x):
        identity = x
        
        x = self.preprocess(x)
        x = self.layers(x)
        
        identity = self.downsampler(identity)
        x = x + identity
        
        x = self.pooling(x)
        return x
    
class SELayer(nn.Module):
    def __init__(self, channels, channel_reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channels, channels // channel_reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channels // channel_reduction, channels),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
        
class Frontend_S(nn.Module):
    """_summary_
    This is frontend of RawformerS
    Args:
        nn (_type_): _description_
    """
    
    def __init__(self, sinc_kernel_size=128, sample_rate=16000):
        """_summary_
        N: number of conv2D-based blocks\\
        N is fixed to 4.
        
        C: output channel of front-end\\
        C is fixed to 64
        
        f: frequency \\
        f is fixed to 23
        
        t: number of temporal bins\\
        for 4 sec, t is 29. for 10 sec, t is 73\\
        
        Args:
            sinc_kernel_size (int, optional): kernel size of sinc layer. Defaults to 128.
            sample_rate (int, optional): _description_. Defaults to 16000.
        """
        super(Frontend_S, self).__init__()
        
        self.sinc_layer = SincConv(in_channels=1, out_channels=70, kernel_size=sinc_kernel_size, sample_rate=sample_rate)
        self.bn = nn.BatchNorm2d(num_features=1) 
        self.selu = nn.SELU(inplace=True)
        
        self.conv_blocks = nn.Sequential(
            Conv2DBlockS(in_channels=1, out_channels=32, is_first_block=True),
            Conv2DBlockS(in_channels=32, out_channels=32),
            Conv2DBlockS(in_channels=32, out_channels=64),
            Conv2DBlockS(in_channels=64, out_channels=64),            
        )
    
    def forward(self, x):
        
        x = x.unsqueeze(dim=1)
        x = self.sinc_layer(x)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.bn(x)
        LFM = self.selu(x)
        
        HFM = self.conv_blocks(LFM)
        
        return HFM
    
class Frontend_L(nn.Module):
    
    def __init__(self, sinc_kernel_size=128, sample_rate=16000):
        """_summary_
        N: number of conv2D-based blocks\\
        N is fixed to 6.
        
        C: output channel of front-end\\
        C is fixed to 64
        
        f: frequency \\
        f is fixed to 23
        
        t: number of temporal bins\\
        for 4 sec, t is 29. for 10 sec, t is 73\\
        
        Args:
            sinc_kernel_size (int, optional): kernel size of sinc layer. Defaults to 128.
            sample_rate (int, optional): _description_. Defaults to 16000.
        """
        super(Frontend_S, self).__init__()
        
        self.sinc_layer = SincConv(in_channels=1, out_channels=70, kernel_size=sinc_kernel_size, sample_rate=sample_rate)
        self.bn = nn.BatchNorm2d(num_features=1) 
        self.selu = nn.SELU(inplace=True)
        
        self.conv_blocks = nn.Sequential(
            Conv2DBlockS(in_channels=1, out_channels=32, is_first_block=True),
            Conv2DBlockS(in_channels=32, out_channels=32),
            Conv2DBlockS(in_channels=32, out_channels=64),
            Conv2DBlockS(in_channels=64, out_channels=64),
            Conv2DBlockS(in_channels=64, out_channels=64), 
            Conv2DBlockS(in_channels=64, out_channels=64),            
        )
    
    def forward(self, x):
        
        x = x.unsqueeze(dim=1)
        x = self.sinc_layer(x)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.bn(x)
        LFM = self.selu(x)
        
        HFM = self.conv_blocks(LFM)
        
        return HFM
        

if __name__ == "__main__":
    from torchinfo import summary
    #model = SincConv(1, 3).to("cuda:0")
    model = Frontend_L().to("cuda:0")
    summary(model, (2, 16000*4))