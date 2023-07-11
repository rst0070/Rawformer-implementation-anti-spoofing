import torch
import torch.nn as nn
import torchaudio
import config
import torch.nn.functional as F
    
class PreEmphasis(nn.Module):
    
    def __init__(self, device, sys_config = config.SysConfig(),exp_config = config.ExpConfig()):
        super(PreEmphasis, self).__init__()
        
        self.pre_emphasis_filter = torch.FloatTensor([[[-exp_config.pre_emphasis, 1.]]]).to(device)
        
        
    def forward(self, x):
        # input shape == (batch, length of utterance)
        # input shape of conv1d should be (batch, 1, length of utterance)
        with torch.no_grad():
            x = x.unsqueeze(1)  
            x = F.pad(input=x, pad=(1, 0), mode='reflect')
            x = F.conv1d(input=x, weight=self.pre_emphasis_filter)

            x = x.squeeze()
            
        return x