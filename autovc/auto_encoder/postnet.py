import torch
import torch.nn as nn
from autovc.utils.net_layers import *


class Postnet(nn.Module):
    """
    Postnet: the last part of figure 3.c in the Paper
        - Five 1-d convolution with 512 channels and kernel size 5
        - 1 layer with in dim 80 and out dim 512
        - 3 layers with in dim = out dim = 512
        - 1 layer with in dim 512 and out dim 80
        - all with tanh activaion funcion
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        """ The first layer """
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )
        """ The 3 midder layers """
        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )
        """ The Final Layer """
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80))
            )

    def forward(self, x):
        """ Takes the output from the decoder module as input """

        """ Through the first 4 layers """
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        """ 
        Out through the 5th and final layer 
            - Out dimension equal to original spectrogram input
        """
        x = self.convolutions[-1](x)

        return x 