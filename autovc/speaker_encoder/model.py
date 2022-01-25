"""
Speaker Identity encoder from https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/master/encoder. See LICENSE.txt
"""

from torch.nn.utils import clip_grad_norm_
from torch import nn
import numpy as np
import torch
from autovc.audio.spectrogram import mel_spec_speaker_encoder
# from autovc.utils.hparams import SpeakerEncoderParams as hparams
from autovc.utils.hparams import SpeakerEncoderParams
import librosa
from autovc.utils import progbar, close_progbar, retrieve_file_paths
import time, wandb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tqdm
import inspect
from autovc.utils import lr_scheduler as lr_schedulers

class SpeakerEncoder(nn.Module):
    """
    The Speaker Encoder module
    input_size: 40 mels
    model_hidden_size = 256
    num_layers = 3

    Not consistenet with AutoVC but the best i could find...
    Trained on GE2E loss for 1.5 M step
    """
    def __init__(self, 
        input_size = SpeakerEncoderParams["model"]["input_size"],
        hidden_size = SpeakerEncoderParams["model"]["hidden_size"],
        embedding_size = SpeakerEncoderParams["model"]["embedding_size"],
        num_layers = SpeakerEncoderParams["model"]["num_layers"],
        batch_first = SpeakerEncoderParams["model"]["batch_first"],
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose = True, 
        ):
        super().__init__()

        
        self.verbose = verbose
        self.device = device if not isinstance(device, str) else torch.device(device)
        self.speakers = {}
        self.logging = {}

        # Network defition
        self.lstm = nn.LSTM(input_size  = input_size,
                            hidden_size = hidden_size,
                            num_layers  = num_layers,
                            batch_first = batch_first
                            ).to(self.device)

        self.linear = nn.Linear(in_features  = hidden_size,
                                out_features = embedding_size
                                ).to(self.device)

        self.relu = torch.nn.ReLU().to(self.device)
        
        
    def do_gradient_ops(self):
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
            
        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def forward(self, utterances, hidden_init=None):
        """
        Computes the embeddings of a batch of utterance spectrograms.
        
        :param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape 
        (batch_size, n_frames, n_channels) 
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers, 
        batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        # Pass the input through the LSTM layers and retrieve all outputs, the final hidden state
        # and the final cell state.
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        # We take only the hidden state of the last layer
        embeds_raw = self.relu(self.linear(hidden[-1]))
        
        # L2-normalize it
        embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
        
        return embeds

    

    
    # def load(self, weights_fpath, device = None):
    #     """
    #     Loads the model in memory. If this function is not explicitely called, it will be run on the 
    #     first call to embed_frames() with the default weights file.
        
    #     :param weights_fpath: the path to saved model weights.
    #     """
    #     device = device if device is not None else self.params.device
    #     try:
    #         checkpoint = torch.load(self.params.model_dir.strip("/") + "/" + weights_fpath, map_location = device)
    #     except:
    #         checkpoint = torch.load(weights_fpath, map_location = device)
    #     self.load_state_dict(checkpoint["model_state"], strict = False)

    #     self.speakers.update(checkpoint.get('speakers', {}))

    #     print("Loaded speaker encoder \"%s\" trained to step %d" % (weights_fpath, checkpoint["step"]))

    def load(self, model_name, model_dir = SpeakerEncoderParams["model_dir"], device = None):
        self.device = device if device is not None else self.device
        # try:
        #     checkpoint = torch.load(self.params.model_dir.strip("/") + "/" + weights_fpath, map_location = self.device)
        # except:
        model_path = model_dir.strip("/") + "/" + model_name
        checkpoint = torch.load(model_path, map_location = self.device)
        self.load_state_dict(checkpoint["model_state"], strict = False)
        self.speakers.update(checkpoint.get('speakers', {}))

        if self.verbose:
            print("Loaded auto encoder \"%s\" trained to step %d" % (model_path, checkpoint["step"]))

    def save(self, model_name, model_dir = SpeakerEncoderParams["learn"]["save_dir"], wandb_run = None):
        # save_name = self.params.model_dir.strip("/") + "/" + self.params.model_name
        model_path = model_dir.strip("/") + "/" + model_name
        torch.save({
            "step": self.logging.get("step"),
            "model_state": self.state_dict(),
            "optimizer_state": self.optimiser.state_dict(),
            "speakers": self.speakers
        }, model_path)

        if wandb_run is not None:
            artifact = wandb.Artifact(model_name, "SpeakerEncoder")
            artifact.add_file(model_path)
            wandb_run.log_artifact(artifact)


    # @stat
    # def load_model(weights_fpath: Path, device=None):
    #     """
    #     Loads the model in memory. If this function is not explicitely called, it will be run on the 
    #     first call to embed_frames() with the default weights file.
        
    #     :param weights_fpath: the path to saved model weights.
    #     :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda"). The 
    #     model will be loaded and will run on this device. Outputs will however always be on the cpu. 
    #     If None, will default to your GPU if it"s available, otherwise your CPU.
    #     """

    #     global _model, _device
    #     if device is None:
    #         _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     elif isinstance(device, str):
    #         _device = torch.device(device)
    #     # _model = SpeakerEncoder(_device, torch.device("cpu"))
    #     _model = SpeakerEncoder(_device)
    #     checkpoint = torch.load(weights_fpath,map_location=torch.device('cpu'))
    #     _model.load_state_dict(checkpoint["model_state"])
    #     _model.eval()
    #     print("Loaded encoder \"%s\" trained to step %d" % (weights_fpath, checkpoint["step"]))
        
    #     return _model
        
    def embed_frames_batch(self, frames_batch):
        """
        Computes embeddings for a batch of mel spectrogram.
        
        :param frames_batch: a batch mel of spectrogram as a numpy array of float32 of shape 
        (batch_size, n_frames, n_channels)
        :return: the embeddings as a numpy array of float32 of shape (batch_size, model_embedding_size)
        """

        frames = torch.from_numpy(frames_batch).to(self.device)
        embed = self.forward(frames).detach().cpu()
        return embed

    
    def embed_utterance(self, wav, using_partials=True, return_partials=False, **kwargs):
        """
        This is the main function!
        Computes an embedding for a single utterance.
        
        # TODO: handle multiple wavs to benefit from batching on GPU
        :param wav: a preprocessed (see utils.py) utterance waveform as a numpy array of float32
        :param using_partials: if True, then the utterance is split in partial utterances of 
        <partial_utterance_n_frames> frames and the utterance embedding is computed from their 
        normalized average. If False, the utterance is instead computed from feeding the entire 
        spectogram to the network.
        :param return_partials: if True, the partial embeddings will also be returned along with the 
        wav slices that correspond to the partial embeddings.
        :param kwargs: additional arguments to compute_partial_splits()
        :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If 
        <return_partials> is True, the partial utterances as a numpy array of float32 of shape 
        (n_partials, model_embedding_size) and the wav partials as a list of slices will also be 
        returned. If <using_partials> is simultaneously set to False, both these values will be None 
        instead.
        """
        # Load the wav from disk if needed
        if isinstance(wav, str):
            wav, source_sr = librosa.load(wav, sr=None)

        # Process the entire utterance if not using partials
        if not using_partials:
            # frames = wav_to_mel_spectrogram(wav)
            frames = mel_spec_speaker_encoder(wav, cut = False)
            embed = self.embed_frames_batch(frames[None, ...])[0]
            if return_partials:
                return embed, None, None
            return embed
        
        # # Compute where to split the utterance into partials and pad if necessary
        # wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
        # max_wave_length = wave_slices[-1].stop
        # if max_wave_length >= len(wav):
        #     wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
        
        # # Split the utterance into partials
        # frames = wav_to_mel_spectrogram(wav)
        # frames_batch = np.array([frames[s] for s in mel_slices])

        frames_batch, wave_slices, _ = mel_spec_speaker_encoder(wav, cut = True, return_slices=True)
        # frames_batch = np.array(frames_batch)
        # partial_embeds = self.embed_frames_batch(frames_batch)
        partial_embeds =  self.embed_frames_batch(np.array(frames_batch))
        
        # Compute the utterance embedding from the partial embeddings
        raw_embed = partial_embeds.mean(axis=0)
        embed = raw_embed / np.linalg.norm(raw_embed, 2)
        
        if return_partials:
            return embed, partial_embeds, wave_slices
        return embed

    def _similarity_matrix(self, embeds):
        '''
        Computes the similarity matrix according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the similarity matrix as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, speakers_per_batch)
        '''

        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / torch.norm(centroids_incl, dim=2, keepdim=True)

        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / torch.norm(centroids_excl, dim=2, keepdim=True)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch).to(self.params.device)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        ## Even more vectorized version (slower maybe because of transpose)
        # sim_matrix2 = torch.zeros(speakers_per_batch, speakers_per_batch, utterances_per_speaker
        #                           ).to(self.device)
        # eye = np.eye(speakers_per_batch, dtype=np.int)
        # mask = np.where(1 - eye)
        # sim_matrix2[mask] = (embeds[mask[0]] * centroids_incl[mask[1]]).sum(dim=2)
        # mask = np.where(eye)
        # sim_matrix2[mask] = (embeds * centroids_excl).sum(dim=2)
        # sim_matrix2 = sim_matrix2.transpose(1, 2)
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix
    
    def _loss(self, embeds):
        '''
        Computes the softmax loss according the section 2.1 of GE2E.
        
        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        '''
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Loss
        sim_matrix = self._similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                         speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.device)
        loss = self.loss_fn(sim_matrix, target)
        
        # EER (not backpropagated)
        # with torch.no_grad():
        #     inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
        #     labels = np.array([inv_argmax(i) for i in ground_truth])
        #     preds = sim_matrix.detach().cpu().numpy()

        #     # Snippet from https://yangcha.github.io/EER-ROC/
        #     fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
        #     eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss#, eer


    def batch_forward(self, batch):
        '''
        Batch of
        
        '''
        return torch.stack([self(b.to(self.params.device)) for b in batch])

   

    def learn(
        self,
        trainloader, 
        n_epochs = SpeakerEncoderParams["learn"]["n_epochs"],
        log_freq = SpeakerEncoderParams["learn"]["log_freq"],
        save_freq = SpeakerEncoderParams["learn"]["save_freq"],
        save_dir = SpeakerEncoderParams["learn"]["save_dir"],
        model_name = SpeakerEncoderParams["learn"]["model_name"],
        wandb_run = None, 
        **opt_params
    ):

        # initialisation
        self.train()
        self.logging["step"] = 0
        self.logging["running_loss"] = 0
        self.logging["log_steps"] = 0
        self.logging["total_time"] = 0
        N_iterations = n_epochs*len(trainloader)
        if wandb_run is not None:
            wandb_run.watch(self, log_freq = log_freq)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(self.device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(self.device)

        # prepare optimiser
        SpeakerEncoderParams["optimizer"].update(opt_params)
        lr_scheduler = SpeakerEncoderParams["optimizer"].pop("lr_scheduler")
        if isinstance(lr_scheduler, str):
            lr_scheduler = lr_schedulers.__dict__.get(lr_scheduler)
        lr_sch_params = {"n_warmup_steps" : SpeakerEncoderParams["optimizer"].pop("n_warmup_steps")}
        self.optimiser = torch.optim.Adam(self.parameters(), **SpeakerEncoderParams["optimizer"])

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(self.optimiser, dim_model = self.linear.out_features, **lr_sch_params)


        
        # begin training
        if self.verbose:
            # print(f"Training Speaker Encoder on {torch.cuda.get_device_name() + ' (cuda)' if 'cuda' in self.params.device else 'cpu'}...")
            progbar(self.logging["step"], N_iterations)
        
        
        for epoch in range(n_epochs):
            self.logging["step_start_time"] = time.time()
            for batch in trainloader:
                embeddings = self.batch_forward(batch)

                loss = self._loss(embeddings)

                # Compute gradients, clip and take a step
                self.optimiser.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = 1) # Clip gradients (avoid exploiding gradients)

                if self.lr_scheduler is not None: 
                    self.lr_scheduler._update_learning_rate()
                self.optimiser.step()

                # update log params
                self.logging["step"] += 1
                self.logging["running_loss"] += loss
                self.logging["log_steps"] += 1

                # print information
                if self.verbose:
                    self.logging["total_time"] += (time.time()-self.logging["step_start_time"])
                    progbar(self.logging["step"], N_iterations, {"sec/step": np.round(self.logging["total_time"]/self.logging["step"])})

                # save model and log to wandb
                if (self.logging["step"] % log_freq == 0 or self.logging["step"] == N_iterations) and wandb_run is not None:
                    self._log(wandb_run, embeddings=embeddings)

                if self.logging["step"] % save_freq == 0 or self.logging["step"] == N_iterations:
                    self.save(model_name, save_dir, wandb_run)

        close_progbar()

    def _log(self, wandb_run, embeddings):
        wandb_run.log({
            "loss" : self.logging["running_loss"]/self.logging["log_steps"]
        }, step = self.logging["step"])
        wandb_run.log({
                'TSNE': self.visualise_embedding(embeddings)
            })
        
        # reset logging 
        self.logging["running_loss"] = 0
        self.logging["log_steps"] = 0


    def visualise_embedding(self, embeddings):
        '''
        embeddings = (number of speakers, number of utterances pr. speaker, lattent dimension)
        
        Returns a TSNE plot with 2 components
        '''
        X = TSNE(n_components = 2).fit_transform(torch.flatten(embeddings, start_dim  = 0, end_dim = 1).detach().cpu().numpy())

        fig, ax = plt.subplots(figsize = (10,10))
        n_speakers = embeddings.size(0)
        n_utterances = embeddings.size(1)
        for speaker in range(n_speakers):
            start = speaker * n_utterances
            stop  = start + n_utterances
            ax.scatter(X[start:stop, 0], X[start:stop,1], alpha = 0.6, zorder = 3)
        ax.grid(ls = '--')

        
        return fig

    def learn_speaker(self, speaker, speaker_path, speaker_path_excluded = []):
        '''
        Learns the mean speaker embedding of a speaker given a speaker and a folder of .wavs of this speaker
        Saves the embedding in self.speakers, which is saved together with the state_dict
        '''
        wav_files = retrieve_file_paths(speaker_path, speaker_path_excluded)

        embeddings = []
        print("Computing speaker embeddings...")
        for wav in tqdm.tqdm(wav_files):
            embeddings.append(self.embed_utterance(wav))
        embeddings = torch.stack(embeddings)
        mean_embedding = embeddings.mean(axis = 0, keepdim = True)

        self.speakers[speaker] = mean_embedding.squeeze()

SpeakerEncoder.learn.__allowed_args__ = inspect.getfullargspec(SpeakerEncoder.learn).args
SpeakerEncoder.learn.__allowed_kw__ = inspect.getfullargspec(torch.optim.Adam).args


if __name__ == "__main__":
    SE = SpeakerEncoder()
    print(SE.linear.out_features)
