from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class SincConv(nn.Module):
    """_summary_
    This code is from https://github.com/clovaai/aasist/blob/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/models/AASIST.py#L325C8-L325C8
    """
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(self,
                 out_channels,
                 kernel_size,
                 sample_rate=16000,
                 in_channels=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 groups=1,
                 mask=False):
        super().__init__()
        if in_channels != 1:

            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (
                in_channels)
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                  (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2*fmax/self.sample_rate) * \
                np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow = (2*fmin/self.sample_rate) * \
                np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = torch.Tensor(np.hamming(
                self.kernel_size)) * torch.Tensor(hideal)

    def forward(self, x, mask=False):
        band_pass_filter = self.band_pass.clone().to(x.device)
        if mask:
            A = np.random.uniform(0, 20)
            A = int(A)
            A0 = random.randint(0, band_pass_filter.shape[0] - A)
            band_pass_filter[A0:A0 + A, :] = 0
        else:
            band_pass_filter = band_pass_filter

        self.filters = (band_pass_filter).view(self.out_channels, 1,
                                               self.kernel_size)

        return F.conv1d(x,
                        self.filters,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=None,
                        groups=1)

class Conv2DBlock_S(nn.Module):
    """__summary__
    This is Conv2DBlock of Rawformer-S.\\
    This block is same as ResNet block of AASIST with some different parameters.
    (https://github.com/clovaai/aasist/blob/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/models/AASIST.py#L413)
    """
    
    def __init__(self, in_channels: int, out_channels: int, is_first_block: bool=False):
        """_summary_

        Args:
            in_channels (int): num of input channels
            out_channels (int): num of output channels
            se_reduction (int, optional): reduction factor for squeeze and excitation of channels. Defaults to 8.
            is_first_block (bool, optional): if this is the first block must be True. Defaults to False.
        """
        
        super(Conv2DBlock_S, self).__init__()
        
        self.normalizer = None
        if not is_first_block:
            self.normalizer = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.SELU(inplace=True)
            )
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 5), padding=(1, 2), stride=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.SELU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(2, 3), padding=(0, 1), stride=1),
        )        
        
        self.downsampler = None
        if in_channels != out_channels:
            self.downsampler = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding=(0, 1), kernel_size=(1, 3), stride=1)
            )
        
        self.pooling = nn.MaxPool2d(kernel_size=(1, 6))
        
    def forward(self, x):
        
        identity = x
        if self.downsampler is not None:
            identity = self.downsampler(identity)
        
        if self.normalizer is not None:
            x = self.normalizer(x)
            
        x = self.layers(x)
        x = x + identity
        
        x = self.pooling(x)
        return x

class Conv2DBlock_L(nn.Module):
    """__summary__
    This is Conv2DBlock of Rawformer-L.\\
    This block is same as ResNet block of AASIST.
    (https://github.com/clovaai/aasist/blob/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/models/AASIST.py#L413)
    """
    
    def __init__(self, in_channels: int, out_channels: int, is_first_block: bool=False):
        """_summary_

        Args:
            in_channels (int): num of input channels
            out_channels (int): num of output channels
            se_reduction (int, optional): reduction factor for squeeze and excitation of channels. Defaults to 8.
            is_first_block (bool, optional): if this is the first block must be True. Defaults to False.
        """
        
        super(Conv2DBlock_L, self).__init__()
        
        self.normalizer = None
        if not is_first_block:
            self.normalizer = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.SELU(inplace=True)
            )
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 3), padding=(1, 1), stride=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.SELU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(2, 3), padding=(0, 1), stride=1),
        )        
        
        self.downsampler = None
        if in_channels != out_channels:
            self.downsampler = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding=(0, 1), kernel_size=(1, 3), stride=1)
            )
        
        self.pooling = nn.MaxPool2d(kernel_size=(1, 3))
        
    def forward(self, x):
        
        identity = x
        if self.downsampler is not None:
            identity = self.downsampler(identity)
        
        if self.normalizer is not None:
            x = self.normalizer(x)
            
        x = self.layers(x)
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
        
class Conv2DBlock_SE(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, scale:int = 8, channel_reduction:int=8):
        super(Conv2DBlock_SE, self).__init__()
        
        self.scale = scale
        self.sub_channels = out_channels // scale
        self.hidden_channels = self.sub_channels * scale
        relu = nn.ReLU(inplace=True)
        
        
        self.normalizer = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.SELU(inplace=True)
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.hidden_channels, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(num_features=self.hidden_channels),
            relu
        )
        
        self.conv2 = []
        for i in range(2, scale+1):
            self.conv2.append(nn.Sequential(
                nn.Conv2d(in_channels=self.sub_channels, out_channels=self.sub_channels, kernel_size=(3, 9), padding=(1, 4)),
                nn.BatchNorm2d(num_features=self.sub_channels),
                relu
            ))
        self.conv2 = nn.ModuleList(self.conv2)
            
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=out_channels, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(num_features=out_channels),
            relu
        )
        
        self.se_module = SELayer(channels=out_channels, channel_reduction=channel_reduction)
        
        self.downsampler = None
        if in_channels != out_channels:
            self.downsampler = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding=(0, 3), kernel_size=(1, 7), stride=1)
            )
            
        self.pooling = nn.MaxPool2d(kernel_size=(1, 6))
        #self.pooling = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 6), padding=(0, 0), )
        
        
    def forward(self, x):
        
        identity = x
        if self.downsampler is not None:
            identity = self.downsampler(identity)
            
        x = self.normalizer(x)
        
        x = self.conv1(x)
        
        x_sub = torch.split(x, split_size_or_sections=self.sub_channels, dim = 1)
        y_sub = [x_sub[0]]
        
        for i in range(1, self.scale):
            y_i = None
            if i == 1:
                y_i = self.conv2[i - 1](x_sub[i])
            else:
                y_i = self.conv2[i - 1](x_sub[i] + y_sub[i-1])
                
            y_sub.append(y_i)
        
        y = torch.cat(y_sub, dim = 1)
        y = self.conv3(y)
        y = self.se_module(y)
        
        y = y + identity
        y = self.pooling(y)
        
        return y
        
        
        
    
class Frontend_S(nn.Module):
    """_summary_
    This is frontend of Rawformer-S
    """
    
    def __init__(self, sinc_kernel_size=128, sample_rate=16000):
        """_summary_
        frontend of Rawformer-S\\
            
        N: number of conv2D-based blocks\\
        N is fixed to 4.
        
        C: output channel of front-end\\
        C is fixed to 64
        
        f: frequency \\
        f is fixed to 23
        
        t: number of temporal bins\\
        for 4 sec, t is 16. for 10 sec, t is 73\\
        
        Args:
            sinc_kernel_size (int, optional): kernel size of sinc layer. Defaults to 128.
            sample_rate (int, optional): _description_. Defaults to 16000.
        """
        super(Frontend_S, self).__init__()
        
        self.sinc_layer = SincConv(in_channels=1, out_channels=70, kernel_size=sinc_kernel_size, sample_rate=sample_rate)
        self.bn = nn.BatchNorm2d(num_features=1) 
        self.selu = nn.SELU(inplace=True)
        
        self.conv_blocks = nn.Sequential(
            Conv2DBlock_S(in_channels=1, out_channels=32, is_first_block=True),
            Conv2DBlock_S(in_channels=32, out_channels=32),
            Conv2DBlock_S(in_channels=32, out_channels=64),
            Conv2DBlock_S(in_channels=64, out_channels=64),            
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
        frontend of Rawformer-L\\
    
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
        super(Frontend_L, self).__init__()
        
        self.sinc_layer = SincConv(in_channels=1, out_channels=70, kernel_size=sinc_kernel_size, sample_rate=sample_rate)
        self.bn = nn.BatchNorm2d(num_features=1) 
        self.selu = nn.SELU(inplace=True)
        
        self.conv_blocks = nn.Sequential(
            Conv2DBlock_L(in_channels=1, out_channels=32, is_first_block=True),
            Conv2DBlock_L(in_channels=32, out_channels=32),
            Conv2DBlock_L(in_channels=32, out_channels=64),
            Conv2DBlock_L(in_channels=64, out_channels=64),
            Conv2DBlock_L(in_channels=64, out_channels=64), 
            Conv2DBlock_L(in_channels=64, out_channels=64),            
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
    
class Frontend_SE(nn.Module):
    """_summary_
    This is frontend of SE-Rawformer
    """
    
    def __init__(self, sinc_kernel_size=128, sample_rate=16000):
        """_summary_
        frontend of Rawformer-S\\
            
        N: number of conv2D-based blocks\\
        N is fixed to 4.
        
        C: output channel of front-end\\
        C is fixed to 64
        
        f: frequency \\
        f is fixed to 23
        
        t: number of temporal bins\\
        for 4 sec, t is 16. for 10 sec, t is 73\\
        
        Args:
            sinc_kernel_size (int, optional): kernel size of sinc layer. Defaults to 128.
            sample_rate (int, optional): _description_. Defaults to 16000.
        """
        super(Frontend_SE, self).__init__()
        
        self.sinc_layer = SincConv(in_channels=1, out_channels=70, kernel_size=sinc_kernel_size, sample_rate=sample_rate)
        self.bn = nn.BatchNorm2d(num_features=1) 
        self.selu = nn.SELU(inplace=True)
        
        self.conv_blocks = nn.Sequential(
            Conv2DBlock_S(in_channels=1, out_channels=32, is_first_block=True),
            Conv2DBlock_SE(in_channels=32, out_channels=32),
            Conv2DBlock_SE(in_channels=32, out_channels=64),
            Conv2DBlock_SE(in_channels=64, out_channels=64),            
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
    model = Frontend_SE().to("cuda:0")
    summary(model, (2, 16000*4))