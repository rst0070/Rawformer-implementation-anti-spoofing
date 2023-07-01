import torch
import torch.nn as nn
from frontend import Frontend_S, Frontend_L
from positional_aggregator import PositionalAggregator1D
from classifier import RawformerClassifier


class Rawformer_S(nn.Module):
    
    def __init__(self, device, sample_rate: int = 16000):
        super(Rawformer_S, self).__init__()
        self.front_end = Frontend_S(sinc_kernel_size=128, sample_rate=sample_rate)
        
        self.positional_embedding = PositionalAggregator1D(max_C = 64, max_ft=73*23, device=device)
        
        self.classifier = RawformerClassifier(C = 64, n_encoder = 2, transformer_hidden=64)# output: [batch, C]
        
    def forward(self, x):        
        x = self.front_end(x)
        x = self.positional_embedding(x)        
        x = self.classifier(x)        
        return x
    
class Rawformer_L(nn.Module):
    
    def __init__(self, device, sample_rate: int = 16000):
        super(Rawformer_L, self).__init__()
        self.front_end = Frontend_L(sinc_kernel_size=128, sample_rate=sample_rate)
        
        self.positional_embedding = PositionalAggregator1D(max_C=64, max_ft=73*23, device=device)
        
        self.classifier = RawformerClassifier(C = 64, n_encoder = 3, transformer_hidden=80)# output: [batch, C]
        
    def forward(self, x):        
        x = self.front_end(x)
        x = self.positional_embedding(x)        
        x = self.classifier(x)        
        return x
    
if __name__ == "__main__":
    from torchinfo import summary
    from fvcore.nn import FlopCountAnalysis
    
    model = Rawformer_S(device="cpu").to("cpu")
    summary(model, (2, 16000*4), device="cpu")
    #flops = FlopCountAnalysis(model=model, inputs=torch.rand((2, 16000*4), device="cuda:0"))
    #print(flops.total())