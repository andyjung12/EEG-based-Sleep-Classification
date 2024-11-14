import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head_num):
        super(MultiHeadAttention, self).__init__()
        self.head_num = head_num
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model) # Query weight
        self.w_k = nn.Linear(d_model, d_model) # Key weight
        self.w_v = nn.Linear(d_model, d_model) # Value weight
        self.w_o = nn.Linear(d_model, d_model) # Output weight
    
    def forward(self, q, k, v, mask=None):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        #split by number of heads
        q = self.split(q)
        k = self.split(k)
        v = self.split(v)
        
        #scale dot product for attention score
        out, attention = self.attention(q,k,v, mask=mask)
        
        # concat and pass to linear layer
        out = self.concat(out)
        out = self.w_o(out)

        return out

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.head_num
        tensor = tensor.view(batch_size, length, self.head_num, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor =tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k ,v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = k.size()

        # dotproduct Query and Key for attention score
        k_t = k.transpose(2, 3) #transpose Key
        score = (q @ k_t) / math.sqrt(d_tensor)

        if mask is not None:
            score = score.masked_fill(mask == 0 ,-e)
        
        score = self.softmax(score) # attention weight

        v = score @ v # attention value

        return v, score

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # -1 은 마지막 차원을 뜻함

        out= (x-mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, head_num, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, head_num=head_num)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
    
    def forward(self, x):
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=None)

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x =self.ffn(x)

        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x