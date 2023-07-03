import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

class ScaledDotProductAttention(nn.Module):
    
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim = -1) # attention score for each query
        
    def forward(self, Q: torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask = None):
        assert len(Q.shape) == len(K.shape) == len(V.shape) == 4 # (batch_size, seq_len, n_head, d_tensor)
        
        d_k = K.shape[3]
        K_t = K.transpose(2, 3)
        score = torch.matmul(Q, K_t) / math.sqrt(d_k)
        score = self.softmax(score)
        V = torch.matmul(score, V)
        
        return V
        
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        
        self.n_head = n_head
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention()
        
        self.W_after_attention = nn.Linear(d_model, d_model)
        
    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask = None):
        assert len(Q.shape) == len(K.shape) == len(V.shape) == 3
        
        Q = self.split_for_multi_head( self.W_Q(Q) )
        K = self.split_for_multi_head( self.W_K(K) )
        V = self.split_for_multi_head( self.W_V(V) )
        
        V = self.attention(Q = Q, K = K, V = V)
        V = self.concat_multi_head(V)
        V = self.W_after_attention(V)
        
        return V
        
    def split_for_multi_head(self, tensor:torch.Tensor):
        assert len(tensor.shape) == 3, f"Tensor should be 3-dimensional, but the size is {tensor.shape}"
        batch, seq_len, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch, seq_len, self.n_head, d_tensor).transpose(1, 2)
        return tensor
    
    def concat_multi_head(self, tensor:torch.Tensor):
        assert len(tensor.shape) == 4
        batch, n_head, seq_len, d_tensor = tensor.size()
        tensor = tensor.transpose(1, 2)
        tensor = tensor.reshape(batch, seq_len, n_head * d_tensor)
        return tensor


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
    
class FFN(nn.Module):
    
    def __init__(self, d_model, ffn_hidden, drop_prob=0.1):
        super(FFN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(ffn_hidden, d_model)
        )
        
    def forward(self, x):
        x = self.network(x)
        return x
    
class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, d_model=64, n_head=8, ffn_hidden=2048, drop_prob=0.1):
        """_summary_
        input size: [batch, seq_length, d_model]
        
        Args:
            d_model (int, optional): dimension of this model.
            the length of each embedding of sequence. Defaults to 64.
            
            n_head (int, optional): number of attention heads. Defaults to 8.
            ffn_hidden (int, optional): hidden size of ffn layer. Defaults to 2048.
            drop_prob (float, optional): _description_. Defaults to 0.1.
        """
        super(TransformerEncoderLayer, self).__init__()
        
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        
        self.dropout1 = nn.Dropout(p = drop_prob)
        self.norm1 = LayerNorm(d_model=d_model)
        
        self.ffn = FFN(d_model=d_model, ffn_hidden=ffn_hidden, drop_prob=drop_prob)
        self.dropout2 = nn.Dropout(p = drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        
    def forward(self, x):
        
        residual = x
        x = self.attention(Q = x, K = x, V = x)
        x = self.dropout1(x)
        x = self.norm1(x + residual)
        
        residual = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual)
        
        return x

class SequencePooling(nn.Module):
    """_summary_
    implementation of SeqPool of below
    Li, Changtao, Feiran Yang, and Jun Yang. 
    "The Role of Long-Term Dependency in Synthetic Speech Detection." 
    IEEE Signal Processing Letters 29 (2022): 1142-1146.
    """
    
    def __init__(self, d_model):
        """_summary_
        [batch_size, ]
        Args:
            d_model (_type_): dimension of transformer encoder. 
            This is the dimension of each element of input sequence.
        """
        super(SequencePooling, self).__init__()
        
        self.linear = nn.Linear(in_features=d_model, out_features=1)
        
    def forward(self, x):
        
        w = self.linear(x)
        w = w.transpose(1, 2)
        w = F.softmax(input=w, dim=-1)
        
        x = torch.matmul(w, x)
        x = x.squeeze()
        return x
    
class RawformerClassifier(nn.Module):
    
    def __init__(self, C:int, n_encoder:int, transformer_hidden:int):
        """_summary_
        [batch_size, f*t, C] -> [batch_size, 2]
        
        Args:
            C (int): number of channels of HFM
            n_encoder (int): number of transformer encoder blocks. (`N` in the paper)
            transformer_hidden (int): 
        """
        super(RawformerClassifier, self).__init__()
        # ------------------- encoder blocks ------------------- #
        self.encoders = OrderedDict()
        for i in range(0, n_encoder):
            self.encoders[f"encoder{i}"] = TransformerEncoderLayer(d_model=C, ffn_hidden=transformer_hidden)
        
        self.encoders = nn.Sequential(self.encoders)
        # ------------------- encoder blocks ------------------- #
        self.seq_pooling = SequencePooling(d_model=C)
        # ------------------- linear layer ------------------- #
        self.linear = nn.Linear(in_features=C, out_features=1)
        
        self.sig = nn.Sigmoid()
        
        
    def forward(self, x):
        
        x = self.encoders(x)
        x = self.seq_pooling(x)
        x = self.linear(x)
        x = self.sig(x)
        x = x.squeeze()
        return x
        
    
if __name__ == "__main__":
    from torchinfo import summary
    #model = SincConv(1, 3).to("cuda:0")
    model = RawformerClassifier(C=64, n_encoder=3, transformer_hidden=80).to("cuda:0")
    # x = torch.ones((1, 2, 2, 2), device="cuda:0")
    # print(x)
    # x = model(x)
    # print(x)
    summary(model, (2, 23*29, 64))
        