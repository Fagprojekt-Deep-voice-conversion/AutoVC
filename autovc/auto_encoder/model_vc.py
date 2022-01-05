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
# from autovc.utils.lr_scheduler import NoamLrScheduler as Noam 
# from autovc.utils.hparams_NEW import 
# from autovc.utils.lr_scheduler import NoamLrScheduler as Noam
# from autovc.utils import lr_scheduler
from autovc.utils.hparams_NEW import AutoEncoder as hparams



class Generator(nn.Module):
    """
    Generator network. The entire thing pieced together (figure 3a and 3c)
    """
    def __init__(self, **params):
        """
        params:
        dim_neck: dimension of bottleneck (set to 32 in the paper)
        dim_emb: dimension of speaker embedding (set to 256 in the paper)
        dim_pre: dimension of the input to the decoder (output of first LSTM layer) (set to 512 in the paper)
        full list of params can be found in `autovc/utils/hparams.py`
        """
        super(Generator, self).__init__()
        # self.freq = freq
        self.params = hparams().update(params)
        self.encoder = Encoder(**self.params.get_collection("Encoder"))
        self.decoder = Decoder(**self.params.get_collection("Decoder"))
        self.postnet = Postnet()

        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.L1Loss()
        self.optimiser = torch.optim.Adam(self.parameters(), **self.params.get_collection("Adam"))
        # self.lr_scheduler = Noam(self.optimiser, d_model = 80, n_warmup_steps = 200)
        self.lr_scheduler = self.params.lr_scheduler(self.optimiser, **self.params.get_collection("lr_scheduler"))

        

    def forward(self, x, c_org, c_trg):
        """
        params:
        x: spectrogram batch dim: (batch_size, time_frames, n_mels (80) )
        c_org: Speaker embedding of source dim (batch size, 256)
        c_trg: Speaker embedding of target (batch size, 256)
        """

        """ Pass x and c_org through encoder and obtain downsampled C1 -> and C1 <- as in figure 3"""
        codes_forward, codes_backward= self.encoder(x, c_org)

        """ 
        If no target provide output the content codes from the content encoder 
        This is for the loss function to easily produce content codes from the final output of AutoVC
        """
        if c_trg is None:
            content_codes= torch.cat([torch.cat(codes_forward, dim = -1), torch.cat(codes_backward, dim = -1)], dim = -1)
            # content_codes = torch.cat([torch.cat(code, dim=-1) for code in codes], dim=-1)
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
        codes_forward_upsampled = torch.cat([c.unsqueeze(-1).expand(-1,-1, self.params.freq) for c in codes_forward], dim = -1)
        last_part = codes_forward[-1].unsqueeze(-1).expand(-1,-1, x.size(-1) - codes_forward_upsampled.size(-1))
        codes_forward_upsampled = torch.cat([codes_forward_upsampled, last_part], dim = -1)

        codes_backward_upsampled = torch.cat([c.unsqueeze(-1).expand(-1,-1, self.params.freq) for c in codes_backward], dim = -1)[:,:,:x.size(-1)]


        """ Concatenates upsampled content codes with target embedding. Dim = (batch_size, 320, input time frames) """
        code_exp = torch.cat([codes_forward_upsampled, codes_backward_upsampled], dim=1)
        encoder_outputs = torch.cat((code_exp,c_trg.unsqueeze(-1).expand(-1,-1,x.size(-1))), dim=1)

        """ Sends concatenate encoder outputs through the decoder """
        mel_outputs = self.decoder(encoder_outputs.transpose(1,2)).transpose(2,1)


        """ Sends the decoder outputs through the 5 layer postnet and adds this output with decoder output for stabilisation (section 4.3)"""
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        """ 
        Prepares final output
        mel_outputs: decoder outputs
        mel_outputs_postnet: decoder outputs + postnet outputs
        contetn_codes: the codes from the content encoder
        """
        mel_outputs = mel_outputs
        mel_outputs_postnet = mel_outputs_postnet
        # content_codes = torch.cat([torch.cat(code, dim = -1) for code in codes], dim = -1)
        content_codes= torch.cat([torch.cat(codes_forward, dim = -1), torch.cat(codes_backward, dim = -1)], dim = -1)
        
        
        return mel_outputs, mel_outputs_postnet, content_codes


    def load_model(self, weights_fpath, device):
        checkpoint = torch.load(weights_fpath, map_location = device)
        self.load_state_dict(checkpoint["model_state"])

        
    def loss(self, X, c_org, out_decoder, out_postnet, content_codes,  mu = 1, lambd = 1):
        """
        Loss function as proposed in AutoVC
        L = Reconstruction Error + mu * Prenet reconstruction Error + lambda * content Reconstruction error
        mu and lambda are set to 1 in the paper.

        params:
            X:              A batch of mel spectograms to convert
            c_org:          Speaker embedding batches of X
            out_decoder:    The output of the decoder (Converted spectogram)
            out_postnet:    The output of the postnet (Converted spectogram)
            content_codes:  The output of the encoder (Content embedding)

        returns the loss function as proposed in AutoVC
        """

        # Create content codes from reconstructed spectogram
        reconstructed_content_codes = self(out_postnet, c_org, None)
       
        # Reconstruction error: 
        #     The mean of the squared p2 norm of (Postnet outputs - Original Mel Spectrograms)
        reconstruction_loss1  = self.criterion1(out_postnet, X)
        
        # Prenet Reconstruction error
        #     The mean of the squared p2 norm of (Decoder outputs - Original Mel Spectrograms)
        reconstruction_loss2 = self.criterion1(out_decoder, X)
        
        # Content reconstruction Error
        #     The mean of the p1 norm of (Content codes of postnet output - Content codes)
        content_loss = self.criterion2(reconstructed_content_codes, content_codes)

        return reconstruction_loss1 + mu * reconstruction_loss2 + lambd * content_loss


    def learn(self, trainloader, n_epochs, lr_scheduler = None, save_every = 1000, models_dir = None , model_path_name = None, **params):
        if torch.cuda.is_available():
            print(f"Training beginning on {torch.cuda.get_device_name(0)}")
        else:
            print(f"Training beginning on cpu")
        step = 0
        self.params = hparams().update(params)
        # ema = 0.9999

        self.train()
        avg_params = self.flatten_params()

        for epoch in range(n_epochs):
            for X, c_org in trainloader:
                # Comutet output using the speaker embedding only of the source
                out_decoder, out_postnet, content_codes = self(X, c_org, c_org)

                # Computes the AutoVC reconstruction loss
                loss = self.loss(X = X, c_org = c_org, out_decoder = out_decoder, out_postnet = out_postnet, content_codes = content_codes)
                
                # Compute gradients, clip and take a step
                self.optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = 1) # Clip gradients (avoid exploiding gradients)

                if self.lr_scheduler is not None: self.lr_scheduler._update_learning_rate()
                self.optimiser.step()

                # Save exponentially smoothed parameters - can be used to avoid too large changes of parameters
                avg_params = self.params.ema_decay * avg_params + (1-self.params.ema_decay) * self.flatten_params()
                step += 1
                print("Step:", step)

                '''
                Add save model stuff and log loss with W&B below.
                To save the exponentially smothed params use self.load_flattenend_params first.
                
                '''

    
        #         if step % 10 == 0:
        #             """ Append current error to L for plotting """
        #             r = error.cpu().detach().numpy()
        #             running_loss.append(r)
        #             pickle.dump(running_loss, open(loss_fpath, "wb"))

        #         if step % save_every == 0:
        #             original_param = flatten_params(model)
        #             load_params(model, avg_params)
        #             print("Saving the model (step %d)" % step)
        #             torch.save({
        #                 "step": step + 1,
        #                 "model_state": model.state_dict(),
        #                 "optimizer_state": optimiser.state_dict(),
        #             }, models_dir + "/" + model_path_name + "average_"+ f"_step{step / 1000}k" ".pt")
        #             load_params(model, original_param)
        #             torch.save({
        #                 "step": step + 1,
        #                 "model_state": model.state_dict(),
        #                 "optimizer_state": optimiser.state_dict(),
        #             }, models_dir + "/" + model_path_name + "_original" +f"_step{step / 1000}k" ".pt")

        #         if step >= n_steps:
        #             break



        # pickle.dump(running_loss, open(loss_fpath, "wb"))
        # print("Saving the model (step %d)" % step)
        # torch.save({
        #     "step": step + 1,
        #     "model_state": model.state_dict(),
        #     "optimizer_state": optimiser.state_dict(),
        # }, models_dir + "/" + model_path_name + "_original" + f"_step{step / 1000}k" ".pt")
        # load_params(model, avg_params)

        # torch.save({
        #     "step": step + 1,
        #     "model_state": model.state_dict(),
        #     "optimizer_state": optimiser.state_dict(),
        # }, models_dir + "/" + model_path_name + "average_" + f"_step{step / 1000}k" ".pt")


    def flatten_params(self):
        '''
        Flattens the parameter to a single vector
        '''
        return torch.cat([param.data.view(-1) for param in self.parameters()], 0)

    def load_flattened_params(self, flattened_params):
        '''
        Loads parameters from flattened params
        '''
        offset = 0
        for param in self.parameters():
            param.data.copy_(flattened_params[offset:offset + param.nelement()].view(param.size()))
            offset += param.nelement()


