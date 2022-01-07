import torch
import torch.nn as nn
import torch.nn.functional as F
from autovc.auto_encoder.net_layers import ConvNorm


class Encoder(nn.Module):
    """
    Content Encoder module as presented in the Paper Figure 3.a

    First the signal is processed through 3 convolutional layers of in_dim = out_dim = 512
    The kernel size is of dim 5x5 and zero padding is performed by padding 2 times.

    After the convolutions the signal is processed through 2 Bidirectional LSTM layers with bottleneck dimension 32
    The BLSTM finally produces a forward output and a backward output of dimension 32 for each timestep (frame) of the input

    The forward input is then downsampled by taking the output at time (31, 63, 95, ...)
    The backward input is then downsampled by taking the output at time (0, 32, 64, ...)

    Notice that the downsampling here is inconsistent in what stated in the Paper... In the Paper it is opposite downsampling for backward and forward
    """
    def __init__(self, dim_neck, dim_emb, freq):
        """
        params:
        dim_neck: Dimension of the bottleneck - set to 32 in the paper
        dim_emb: Dimension of the speaker embedding - set to 256 in the paper
        freq: sampling frequency for downsampling - set to 32 in the paper
        """
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        
        convolutions = []
        """ The 3 convolutional layers"""
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80+dim_emb if i==0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        """ The 2 BLSTM layers """
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org):
        """
        Process spectrogram of dim (batch_size, time_frames, n_mels) and the speaker embedding of dim 256
        n_mels is set to be 80.

        params:
        x: mel spectrogram
        c_org: embedding of speaker
        """

        """ Concatenates Spectrogram and speaker embedding"""
        # x = x.squeeze(1).transpose(2,1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)

        """ Process through convolutional layers with ReLu activation function """
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        """ Process through BLSTM layers and obtain forward and backward output"""
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]

        """
        Donwsampling...
        These lines are from the git repo  https://github.com/auspicious3000/autovc but only functional for certain inputs...
        """
        #codes = []
        #for i in range(0, outputs.size(1), self.freq):
        #    print(i)
        #    codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))


        """
        Downsampling...
        Slight Adjustments to be consistent with the paper
        Down sampling as in figure 3, box e and f
        """
        codesA = [out_forward[:, i, :] for i in range(self.freq-1, outputs.size(1), self.freq)]  # (31, 63, 95, ...)
        codesB = [out_backward[:, i, :] for i in range(0, outputs.size(1), self.freq)]  # (0, 32, 64, ... )

        return codesA, codesB