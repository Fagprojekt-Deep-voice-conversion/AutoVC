"""
AutoVC model from https://github.com/auspicious3000/autovc. See LICENSE.txt

The model has the same architecture as proposed in "AutoVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss" - referred to as 'the Paper'
Everything is shown in figure 3 in the Paper - please have this by hand when reading through
"""

import torch
import torch.nn as nn
import wandb

from autovc.auto_encoder.net_layers import *
from autovc.auto_encoder.encoder import Encoder
from autovc.auto_encoder.decoder import Decoder
from autovc.auto_encoder.postnet import Postnet
# from autovc.utils.audio import get_mel_frames
# from autovc.utils.hparams import AutoEncoderParams as hparams
from autovc.utils.hparams_new import AutoEncoderParams
from autovc.utils import progbar, close_progbar
import time
import numpy as np
from torch.nn.functional import pad
import matplotlib.pyplot as plt


class AutoEncoder(nn.Module):
    """
    Generator network. The entire thing pieced together (figure 3a and 3c)
    """
    def __init__(self, 
        dim_neck = AutoEncoderParams["model"]["dim_neck"],
        dim_emb = AutoEncoderParams["model"]["dim_emb"],
        dim_pre = AutoEncoderParams["model"]["dim_pre"],
        freq = AutoEncoderParams["model"]["freq"],
        verbose = True, 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ):
        """
        Parameters
        ----------
                dim_neck: 
            Dimension of bottleneck (set to 32 in the paper)
        dim_emb: 
            Dimension of speaker embedding (set to 256 in the paper)
        dim_pre: 
            Dimension of the input to the decoder (output of first LSTM layer) (set to 512 in the paper)
        freq: 
            Sampling frequency for downsampling - set to 32 in the paper
        verbose: 
            Whether to output information in terminal
        device:
            A torch device to store the model on
            If string, the value is passed to torch.device
        """
        super(AutoEncoder, self).__init__()
    
        self.verbose = verbose
        self.device = device if not isinstance(device, str) else torch.device(device)
        self.logging = {}

        # self.params = AutoEncoderParams["model"].update(params)
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()
        

    def forward(self, x, c_org, c_trg):
        """
        Parameters
        ----------
        x: 
            Spectrogram batch
            dim: (batch_size, time_frames, n_mels (80) )
        c_org: 
            Speaker embedding of source 
            dim: (batch size, 256)
        c_trg: Speaker embedding of target 
            dim: (batch size, 256)

        Return
        ------
        mel_outputs: 
            The converted output
        mel_outputs_postnet: 
            The refined (by postnet) out put
        content_codes: 
            The content vector - the content encoder output
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
        codes_forward_upsampled = torch.cat([c.unsqueeze(-1).expand(-1,-1, self.encoder.freq) for c in codes_forward], dim = -1)
        last_part = codes_forward[-1].unsqueeze(-1).expand(-1,-1, x.size(-1) - codes_forward_upsampled.size(-1))
        codes_forward_upsampled = torch.cat([codes_forward_upsampled, last_part], dim = -1)

        codes_backward_upsampled = torch.cat([c.unsqueeze(-1).expand(-1,-1, self.encoder.freq) for c in codes_backward], dim = -1)[:,:,:x.size(-1)]


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
        content_codes: the codes from the content encoder
        """
        mel_outputs = mel_outputs
        mel_outputs_postnet = mel_outputs_postnet
        # content_codes = torch.cat([torch.cat(code, dim = -1) for code in codes], dim = -1)
        content_codes= torch.cat([torch.cat(codes_forward, dim = -1), torch.cat(codes_backward, dim = -1)], dim = -1)
        
        
        return mel_outputs, mel_outputs_postnet, content_codes


    def load(self, model_name, model_dir = AutoEncoderParams["learn"]["model_dir"], device = None):
        self.device = device if device is not None else self.device
        # try:
        #     checkpoint = torch.load(self.params.model_dir.strip("/") + "/" + weights_fpath, map_location = self.device)
        # except:
        model_path = model_dir.strip("/") + "/" + model_name
        checkpoint = torch.load(model_path, map_location = self.device)
        self.load_state_dict(checkpoint["model_state"])
        if self.verbose:
            print("Loaded auto encoder \"%s\" trained to step %d" % (model_path, checkpoint["step"]))

    def save(self, model_name, model_dir = AutoEncoderParams["learn"]["model_dir"], wandb_run = None):
        model_path = model_dir.strip("/") + "/" + model_name
        torch.save({
            "step": self.logging.get("step"),
            "model_state": self.state_dict(),
            "optimizer_state": self.optimiser.state_dict(),
        }, model_path)

        if wandb_run is not None:
            artifact = wandb.Artifact(model_name, "AutoEncoder")
            artifact.add_file(model_path)
            wandb_run.log_artifact(artifact)

        
    def _loss(self, X, c_org, out_decoder, out_postnet, content_codes,  mu = 1, lambd = 1):
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


    def learn(self, 
        trainloader, 
        n_epochs, 
        log_freq = AutoEncoderParams["learn"]["log_freq"],
        save_freq = AutoEncoderParams["learn"]["save_freq"],
        model_dir = AutoEncoderParams["learn"]["model_dir"],
        model_name = AutoEncoderParams["learn"]["model_name"],
        example_freq = AutoEncoderParams["learn"]["example_freq"],
        # example_sources = ...
        ema_decay = AutoEncoderParams["learn"]["ema_decay"],
        wandb_run = None, 
        **opt_params
    ):
        """
        Method for training the auto encoder

        Params
        ------
        The most important parameters to know are the following and a full list can be found in `autovc/utils/hparams.py`

        trainloader:
            a data loader containing the training data
        n_epochs:
            how many epochs to train the model for
        ema_decay
        log_freq
        save_freq
        model_name
        model_dir
        wandb_run
        **opt_params:
            kwargs are given to the optimizer
            If 'lr_scheduler' and 'n_warmup_steps' are specified, these are used to construct a learning rate scheduler.
        """

        # initialisation
        self.logging["step"] = 0
        self.logging["running_loss"] = 0
        self.logging["log_steps"] = 0
        self.logging["total_time"] = 0
        N_iterations = n_epochs*len(trainloader)
        self.train()
        avg_params = self.flatten_params()
        if wandb_run is not None:
            wandb_run.watch(self, log_freq = log_freq)

        # prepare optimiser
        opt_params = AutoEncoderParams["learn"]["optimizer"].update(opt_params)
        lr_scheduler = opt_params.pop("lr_scheduler")
        lr_sch_params = {"n_warmup_steps" : opt_params.pop("n_warmup_steps")}

        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.L1Loss()
        self.optimiser = torch.optim.Adam(self.parameters(), **opt_params)
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(self.optimiser, dim_model = AutoEncoderParams["audio"]["num_mels"], **lr_sch_params)

        

        # begin training
        if self.verbose:
            print(f"Training Auto Encoder on {torch.cuda.get_device_name() + ' (cuda)' if 'cuda' in self.device else 'cpu'}...")
            progbar(self.logging["step"], N_iterations)
        
        for epoch in range(n_epochs):
            step_start_time = time.time()
            for X, c_org in trainloader:
                # Compute output using the speaker embedding only of the source
                out_decoder, out_postnet, content_codes = self(X, c_org, c_org)

                # Computes the AutoVC reconstruction loss
                loss = self._loss(X = X, c_org = c_org, out_decoder = out_decoder, out_postnet = out_postnet, content_codes = content_codes)
                
                # Compute gradients, clip and take a step
                self.optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = 1) # Clip gradients (avoid exploiding gradients)

                if self.lr_scheduler is not None: 
                    self.lr_scheduler._update_learning_rate()
                self.optimiser.step()

                # Save exponentially smoothed parameters - can be used to avoid too large changes of parameters
                avg_params = ema_decay * avg_params + (1-ema_decay) * self.flatten_params()
                

                # update log params
                self.logging["step"] += 1
                self.logging["running_loss"] += loss
                self.logging["log_steps"] += 1

                # print information
                if self.verbose:
                    self.logging["total_time"] += (time.time()-step_start_time)
                    progbar(self.logging["step"], N_iterations, {"sec/step": np.round(self.logging["total_time"]/self.logging["step"])})

                
                

                # save model and log to wandb - To save the exponentially smothed params use self.load_flattenend_params first.
                if (self.logging["step"] % log_freq == 0 or self.logging["step"] == N_iterations) and wandb_run is not None:
                    self._log(wandb_run, X, out_postnet)

                if self.logging["step"] % save_freq == 0 or self.logging["step"] == N_iterations:
                    self.save(model_name, model_dir, wandb_run)
                    
                    
                      
        if self.verbose: close_progbar()

    def _log(self, wandb_run, X, out_postnet):
        wandb_run.log({
            "loss" : self.logging["running_loss"]/self.logging["log_steps"]
        }, step = self.logging["step"])
        wandb_run.log({
                'Conversion': self.plot_conversion(X[0], out_postnet[0])
            })
        plt.close()
        self.logging["running_loss"] = 0
        self.logging["log_steps"] = 0 

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


    def batch_forward(self, batch, c_org, c_trg, overlap = 0.5):
        '''
        Converts a batch of mel frames and pastes them togehther again.

        params:
            batch: (batch_size, 80 (mels), partial_utterance_n_frames)
                    here the batch size is the result of chopping the spectogram in frames of size 'partial_utterance_n_frames'.

            c_org: the speaker embedding of source (1, 256)
            c_trg: the speaker embedding of target (1, 256)
            overlap: hov much each frame in batch overlaps
        
        returns:
            A conversion, where each frame in batch is converted independently and pasted togehter afterwards
        '''
        
        # mel_frames = get_mel_frames(wav,
        #                             audio_to_melspectrogram,
        #                             min_pad_coverage=0.75,
        #                             order = 'MF',
        #                             sr = 22050, 
        #                             mel_window_step = 12.5, 
        #                             partial_utterance_n_frames = 250,
        #                             overlap = overlap,
                                    
        # batch = torch.stack(mel_frames)

        # Expand embeddings to match the batch size and convert voice
        c_org = c_org.expand(batch.size(0),-1)
        c_trg = c_trg.expand(batch.size(0),-1)
        _, output, _ = self(batch, c_org, c_trg)

        
        N = batch.size(-1)

        # Pad with nans, and take mean of converted frames.
        frames = list(output)
        M = len(frames)
        T = int(N * (1-overlap))
        for i in range(M):
            frames[i] = pad(frames[i], (i * T, (M-i-1) * T) , mode = 'constant', value = torch.nan)
        X = torch.stack(frames)
        return X.nanmean(axis = 0)

    def plot_conversion(self, original, converted):
        fig, ax = plt.subplots(ncols = 2, figsize = (20,10))

        original = original.detach().cpu().numpy()
        converted = converted.detach().cpu().numpy()
        ax[0].matshow(original)
        ax[0].set_title("Original")
        ax[1].matshow(converted)
        ax[1].set_title("Reconstructed")


        return fig


if __name__ == "__main__":
    AE = AutoEncoder()
    print(AE.postnet.parameters)

        
        

        
        

        
