import torch
import torch.nn as nn
import torch.nn.functional as F
from autovc.utils.net_layers import ConvNorm, LinearNorm
        
class Decoder(nn.Module):
    """
    Decoder module as proposed in the Paper figure 3.c
    The Decoder takes a input of the upsampled encoder outputs concatenated with the speaker embedding
    Input dim = 32 * 2 + 256 = 320

    First the input is process through a single LSTM layer (inconsistent with paper!)

    Secondly the signal is processed through 3 convolutional layers of in_dim = out_dim = 512
    The kernel size is of dim 5x5 and zero padding is performed by padding 2 times.
        (as in the encoder)

    Afterwards the signal i processed through 2 LSTM layers with out dimension 1024 (in the paper it's 3 layers...)

    Finally a linear projection to dim 80 is performed.
    The output has the same dimensions as the mel input (batch_size, time_frames, n_mels (80) )

    """
    def __init__(self, dim_neck, dim_emb, dim_pre):
        """
        params:
        dim_neck: the bottleneck dimension (set to 32 in the paper)
        dim_emb: speaker embedding dimension (set to 256 in the paper)
        dim_pre: out dimension of the pre LSTM layer (set to 512 in paper)
        """
        super(Decoder, self).__init__()

        """ The pre-LSTM layer. In: 320, out: 512 """
        self.lstm1 = nn.LSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True)

        """ 3 convolutional layers with batch Normalisation """
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        """ Secondary double LSTM layer """
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)

        """ Final linear projection to original dimension """
        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):
        """
        params:
        x: dimension 320. C1 <- + C1 -> + Speaker embedding
        """
        """ Sends signal through the first LSTM layer """
        #self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)

        """ Sends signal through the convolutional layers with batch normalisation and ReLu activation """
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        """ Through the secondary double LSTM layer """
        outputs, _ = self.lstm2(x)

        """ Final projection unto original dimension """
        decoder_output = self.linear_projection(outputs)

        return decoder_output   