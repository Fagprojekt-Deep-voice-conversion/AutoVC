"""
AutoVC model from https://github.com/auspicious3000/autovc. See LICENSE.txt

The model has the same architecture as proposed in "AutoVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss" - referred to as 'the Paper'
Everything is shown in figure 3 in the Paper - please have this by hand when reading through
"""

import torch
import torch.nn as nn
from autovc.utils.net_layers import *
from autovc.auto_encoder.encoder import Encoder
from autovc.auto_encoder.decoder import Decoder
from autovc.auto_encoder.postnet import Postnet
   

class Generator(nn.Module):
    """
    Generator network. The entire thing pieced together (figure 3a and 3c)
    """
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        """
        params:
        dim_neck: dimension of bottleneck (set to 32 in the paper)
        dim_emb: dimension of speaker embedding (set to 256 in the paper)
        dim_pre: dimension of the input to the decoder (output of first LSTM layer) (set to 512 in the paper)
        """
        super(Generator, self).__init__()
        self.freq = freq
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()

    def forward(self, x, c_org, c_trg):
        """
        params:
        x: spectrogram batch dim: (batch_size, time_frames, n_mels (80) )
        c_org: Speaker embedding of source dim (batch size, 256)
        c_trg: Speaker embedding of target (batch size, 256)
        """

        """ Pass x and c_org through encoder and obtain downsampled C1 -> and C1 <- as in figure 3"""
        codes = self.encoder(x, c_org)

        """ 
        If no target provide output the content codes from the content encoder 
        This is for the loss function to easily produce content codes from the final output of AutoVC
        """
        if c_trg is None:
            content_codes = torch.cat([torch.cat(code, dim=-1) for code in codes], dim=-1)
            return content_codes

        """ 
        Upsampling as in figure 3e-f.
        Recall the forward output of the decoder is downsampled at time (31, 63, 95, ...) and the backward output at (0, 32, 64, ...)
        The upsampling copies the downsampled to match the original input.
        E.g input of dim 100:
            Downsampling
            - Forward: (31, 63, 95)
            - Backward: (0, 32, 64, 96)
            Upsampling:
            - Forward: (0-31 = 31, 32-63 = 63, 64-100 = 95)
            - Backward: (0-31 = 0, 32-63 = 32, 64-95 = 64, 96-100 = 96)
        """
        tmp = []
        for code in codes:
            L = len(code)
            diff = x.size(1) - L * self.freq
            Up_Sampling = [sample.unsqueeze(1).expand(-1, self.freq + bool( (i+1) == L) * diff, -1) for i, sample in enumerate(code)]
            tmp.append(torch.cat(Up_Sampling, dim = 1))

        """ Concatenates upsampled content codes with target embedding. Dim = (batch_size, input time frames, 320) """
        code_exp = torch.cat(tmp, dim=-1)
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)

        """ Sends concatenate encoder outputs through the decoder """
        mel_outputs = self.decoder(encoder_outputs)


        """ Sends the decoder outputs through the 5 layer postnet and adds this output with decoder output for stabilisation (section 4.3)"""
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)

        """ 
        Prepares final output
        mel_outputs: decoder outputs
        mel_outputs_postnet: decoder outputs + postnet outputs
        contetn_codes: the codes from the content encoder
        """
        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)
        content_codes = torch.cat([torch.cat(code, dim = -1) for code in codes], dim = -1)

        return mel_outputs, mel_outputs_postnet, content_codes




