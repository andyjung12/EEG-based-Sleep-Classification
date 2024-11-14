import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from transformer import *
        
class TCNblock(nn.Module):
    def __init__(self, in_chan, chan, out_chan, kernel_size, stride, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = weight_norm(nn.Conv1d(in_chan, chan, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(chan, out_chan, kernel_size=kernel_size, stride=stride, dilation=dilation))
        self.act = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dimension_expand = nn.Conv1d(in_chan, out_chan, kernel_size=1, stride=1)
        self.init_weights()

    def zero_pad(self, batch, channel, kernel_size, dilation):
        pad = torch.zeros(batch, channel, (kernel_size-1)*dilation)
        return pad
    def init_weights(self):
            self.conv1.weight.data.normal_(0, 0.01)
            self.conv2.weight.data.normal_(0, 0.01)
                
    def forward(self, x):
        input = self.dimension_expand(x)
        B, C, D = x.shape
        
        zero_pad = self.zero_pad(B, C, self.kernel_size, self.dilation).to(input.device)
        x = torch.cat([zero_pad, x], dim=2)
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout1(x)

        B, C, D = x.shape
        zero_pad = self.zero_pad(B, C, self.kernel_size, self.dilation).to(input.device)
        x = torch.cat([zero_pad, x], dim=2)
        x = self.conv2(x)
        x = self.act(x)
        x = self.dropout2(x)
        x = x + input
        
        return x

class TemporalConvNet(nn.Module):
    def __init__(self, in_chan, chan, out_chan, dilation):
        super().__init__()
        if len(dilation) == 2:
            self.tcnblocks = nn.Sequential(TCNblock(in_chan, chan, chan, kernel_size=3, stride=1, dilation=dilation[0]),
                                           TCNblock(chan, chan, out_chan, kernel_size=3, stride=1, dilation=dilation[1],)
            )
        
        if len(dilation) == 3:
            self.tcnblocks = nn.Sequential(TCNblock(in_chan, chan, chan, kernel_size=3, stride=1, dilation=dilation[0]),
                                        TCNblock(chan, chan, out_chan, kernel_size=3, stride=1, dilation=dilation[1]),
                                        TCNblock(chan, chan, out_chan, kernel_size=3, stride=1, dilation=dilation[2]),
                                                 )
        if len(dilation) == 4:
            self.tcnblocks = nn.Sequential(TCNblock(in_chan, chan, chan, kernel_size=3, stride=1, dilation=dilation[0]),
                                        TCNblock(chan, chan, out_chan, kernel_size=3, stride=1, dilation=dilation[1]),
                                        TCNblock(chan, chan, out_chan, kernel_size=3, stride=1, dilation=dilation[2]),
                                        TCNblock(chan, chan, out_chan, kernel_size=3, stride=1, dilation=dilation[2]),
                                                 )
 
        
    def forward(self, x):
        BL, F, T = x.shape  
        # x = x.reshape(B, L*F).unsqueeze(1)
        x = self.tcnblocks(x)

        return x

class AttentionWeightSum(nn.Module):
    def __init__(self, in_dim, attention_size):
        super().__init__()
        self.in_dim = in_dim

        self.wa = nn.Linear(in_dim, attention_size, bias=True)
        self.tanh = nn.Tanh()
        self.ae = nn.Linear(attention_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, inputs):
        # inputs.shape = (BL, C, D)
        weight = self.wa(inputs) # weight.shape = (BL, C, A)
        weight = self.tanh(weight) # weight.shape = (BL, C, A)
        weight = self.ae(weight) # weight.shape = (BL, C, 1)
        weight = self.softmax(weight)
        output = inputs * weight # output.shape = (BL, C, D)
        output = output.sum(dim=-2) # output.shape = (BL, D)
        return output

class TimeFeatureExtractor(nn.Module):
    def __init__(self, out_chan):
        super(TimeFeatureExtractor, self).__init__()

        # Convolutional layer for Delta band (0–4 Hz)
        self.conv_delta = nn.Sequential(
                                        nn.Conv1d(in_channels=1, out_channels=out_chan, kernel_size=25, stride=6),
                                        nn.BatchNorm1d(out_chan),
                                        nn.ReLU(),
                                        nn.MaxPool1d(kernel_size=8, stride=8),
        )

        # Convolutional layer for Theta (4–8 Hz)
        self.conv_theta = nn.Sequential(
                                        nn.Conv1d(in_channels=1, out_channels=out_chan, kernel_size=13, stride=3),
                                        nn.BatchNorm1d(out_chan),
                                        nn.ReLU(),
                                        nn.MaxPool1d(kernel_size=16, stride=16),
        )
        # Convolutional layer for Theta (8–12 Hz)
        self.conv_alpha = nn.Sequential(
                                        nn.Conv1d(in_channels=1, out_channels=out_chan, kernel_size=9, stride=2),
                                        nn.BatchNorm1d(out_chan),
                                        nn.ReLU(),
                                        nn.MaxPool1d(kernel_size=16, stride=16),
        )
        # Convolutional layer for Theta (8–12 Hz)
        self.conv_sigma = nn.Sequential(
                                        nn.Conv1d(in_channels=1, out_channels=out_chan, kernel_size=7, stride=2),
                                        nn.BatchNorm1d(out_chan),
                                        nn.ReLU(),
                                        nn.MaxPool1d(kernel_size=16, stride=16),
        )
        
        self.conv_beta = nn.Sequential(
                                        nn.Conv1d(in_channels=1, out_channels=out_chan, kernel_size=4, stride=1),
                                        nn.BatchNorm1d(out_chan),
                                        nn.ReLU(),
                                        nn.MaxPool1d(kernel_size=32, stride=32),
        )

        self.attention = MultiHeadAttention(d_model=128, head_num=4)
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)

    
    def forward(self, x):
        # x.shape = [BL, 1, 3000]
        delta = self.global_average_pool(self.conv_delta(x)).unsqueeze(1)
        theta = self.global_average_pool(self.conv_theta(x)).unsqueeze(1)
        alpha = self.global_average_pool(self.conv_alpha(x)).unsqueeze(1)
        sigma = self.global_average_pool(self.conv_sigma(x)).unsqueeze(1)
        beta = self.global_average_pool(self.conv_beta(x)).unsqueeze(1)
        
        features = torch.cat([delta, theta, alpha, sigma, beta], dim=1) # BL, 5, 128
        features = features.reshape(-1, 5, 128)
        features = self.attention(features, features, features) # BL, 5, 128
        features = features.sum(dim=1)
            
        return features

class Model(nn.Module):
    def __init__(self, device, out_chan=128):
        super().__init__()
            
        # Feature Extraction
        self.raw_FE = TimeFeatureExtractor(out_chan)
        # Time network stream
        self.inter_backward_time = TemporalConvNet(in_chan=128, chan=64, out_chan=64, dilation=[1,2,4])
        self.inter_forward_time = TemporalConvNet(in_chan=128, chan=64, out_chan=64, dilation=[1,2,4])
        # T-F network stream    
        self.intra_backward_stft = TemporalConvNet(in_chan=129, chan=64, out_chan=64, dilation=[1,2,4])
        self.intra_forward_stft = TemporalConvNet(in_chan=129, chan=64, out_chan=64, dilation=[1,2,4])
        self.inter_backward_stft = TemporalConvNet(in_chan=128, chan=64, out_chan=64, dilation=[1,2,4])
        self.inter_forward_stft = TemporalConvNet(in_chan=128, chan=64, out_chan=64, dilation=[1,2,4])
        
        # Attention
        self.time_attention = MultiHeadAttention(d_model=128, head_num=4)

        # Classifier
        self.joint_classifier = nn.Linear(256, 5)
        self.raw_classifier = nn.Linear(128, 5)
        self.tf_classifier = nn.Linear(128, 5)

    def forward(self, raw, stft):
        B, L, C, D = raw.shape
        _, _, C, F, T = stft.shape
        
        ####### Raw ########
        raw = raw.reshape(-1, C, D)
        raw = self.raw_FE(raw) # BL, 2208
        raw = raw.reshape(B, L, -1).permute(0, 2, 1).contiguous()
        
        forward_raw = raw
        backward_raw = torch.flip(raw, dims=[2])
        forward_raw = self.inter_forward_time(forward_raw)
        backward_raw = self.inter_backward_time(backward_raw)
        backward_raw = torch.flip(backward_raw, dims=[2])
        raw = torch.cat([forward_raw, backward_raw], dim=1) # B, 656, L
        raw = raw.permute(0, 2, 1).contiguous()        

        ####### STFT #######
        stft = stft.reshape(-1, F, T)
        forward_stft = stft
        backward_stft = torch.flip(stft, dims=[2])
        forward_stft = self.intra_forward_stft(forward_stft)
        backward_stft = self.intra_backward_stft(backward_stft)
        backward_stft = torch.flip(backward_stft, dims=[2])
        stft = torch.cat([forward_stft, backward_stft], dim=1) # BL, 128, T
        stft = stft.permute(0, 2, 1).contiguous()
        stft = self.time_attention(stft, stft, stft).sum(dim=1)
        stft = stft.reshape(B, L, -1).permute(0, 2, 1).contiguous()

        forward_stft = stft
        backward_stft = torch.flip(forward_stft, dims=[2])
        forward_stft = self.inter_forward_stft(forward_stft)
        backward_stft = self.inter_backward_stft(backward_stft)
        backward_stft = torch.flip(backward_stft, dims=[2])
        stft = torch.cat([forward_stft, backward_stft], dim=1)
        stft = stft.permute(0, 2, 1).contiguous() # B, L, 128        

        joint = torch.cat([raw, stft], dim=-1)
        joint = self.joint_classifier(joint)
        raw = self.raw_classifier(raw)
        stft = self.tf_classifier(stft)

        
        return raw, stft, joint

