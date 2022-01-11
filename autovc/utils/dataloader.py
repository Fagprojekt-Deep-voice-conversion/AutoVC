from ctypes import ArgumentError
import numpy as np 
import torch
import os
from torch.utils.data import DataLoader, Dataset
from autovc.utils.audio import audio_to_melspectrogram
from autovc.speaker_encoder.model import SpeakerEncoder
from autovc.speaker_encoder.utils import *
from torch.nn.functional import pad
from autovc.utils.audio import remove_noise
from autovc.utils.progbar import close_progbar, progbar
from autovc.utils.core import retrieve_file_paths
from autovc.utils.hparams import WaveRNNParams 

vocoder_params = WaveRNNParams()
class TrainDataLoader(Dataset):
    '''
    A Data Loader class for training AutoVC.
    Takes a path to a folder with data (.wav files) and makes a generator object (data loader) 
    which batches spectorgrams of utterances and the corresponding speaker identiy embeddings.
    
    If spectograms in batch are of unequal size the smaller are padded with zeros to match the size of the largest.
    '''

    def __init__(self, speaker_encoder, data_path = None, wav_files = None,  chop = False):
        super(TrainDataLoader, self).__init__()

        # Load wav files. Create spectograms and embeddings
        if wav_files is not None:
            self.wav_files = wav_files
        elif data_path is not None:
            self.wav_files = retrieve_file_paths(data_path)
        else:
            raise ArgumentError(f"Either data_path or wav_files must be different from None")
                
        self.mel_spectograms, self.embeddings, N = [], [], len(self.wav_files)

        print("Creating mel spectrograms and embeddings...")
        progbar(0, N)
        for i, wav in enumerate(self.wav_files):
            # make nice sounds

            if chop:
                mel_frames = get_mel_frames(wav, audio_to_melspectrogram, order = 'MF', sr = vocoder_params.sample_rate, mel_window_step = vocoder_params.mel_window_step, partial_utterance_n_frames = 160 )
                embeds     = speaker_encoder.embed_utterance(wav)
                self.mel_spectograms.extend(mel_frames)
                self.embeddings.extend([embeds.clone() for _ in range(len(mel_frames))])
            else:
                mel_frames = torch.from_numpy(audio_to_melspectrogram(wav))
                embeds     = speaker_encoder.embed_utterance(wav)
                self.mel_spectograms.append(mel_frames)
                self.embeddings.append(embeds)
            progbar(i+1, N)
        close_progbar()

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, index):
        return self.mel_spectograms[index], self.embeddings[index]


    def collate_fn(self, batch):
        '''
        Pad with zeros if spectrograms are of unequal sizes.
        '''
        spectrograms, embeddings = zip(*batch)
        max_length = max([spectogram.size(-1) for spectogram in spectrograms]) 
        padded_spectrograms = [pad(spectogram, (0, max_length - spectogram.size(-1))) for spectogram in spectrograms]

        return torch.stack(padded_spectrograms), torch.stack(embeddings)

    def get_dataloader(self, batch_size=1, shuffle=False,  num_workers=0, pin_memory=False):
        return torch.utils.data.DataLoader(
            self,  
            batch_size=batch_size, 
            num_workers= num_workers, 
            shuffle=shuffle,
            collate_fn = self.collate_fn
        )


class SpeakerEncoderDataLoader(Dataset):
    def __init__(self, data_dict):
        super(SpeakerEncoderDataLoader, self).__init__()

        # Find wav files in dictionary of data        
        wav_files = [sum([retrieve_file_paths(data_dir_path)[:10] for data_dir_path in speaker_data_dir], []) for speaker_data_dir in data_dict.values()]

        # Compute mel spectograms
        speakers = len(data_dict.keys())
        N = sum([len(d) for d in wav_files])
        self.datasets = [[] for _ in wav_files]
        t = 0
        print("Creating mel spectograms ...")
        progbar(t, N)
        for i in range(speakers):
            for wav in wav_files[i]:
                # Compute where to split the utterance into partials and pad if necessary
                frames_batch = get_mel_frames(wav, wav_to_mel_spectrogram )
                
                self.datasets[i].extend(frames_batch)
                t += 1
                progbar(t, N)
        
        print(f"The datasets are of lengths: {[len(d) for d in self.datasets]}")

    def __getitem__(self, i):
        return tuple(d[i % len(d)] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)

    def collate_fn(self, batch):

        return torch.stack(tuple(torch.stack(b) for b in list(zip(*batch)) ))

    def get_dataloader(self, batch_size=2, shuffle=False,  num_workers=0, pin_memory=False, **kwargs):
        return torch.utils.data.DataLoader(
            self,  
            batch_size      = batch_size, 
            num_workers     = num_workers, 
            shuffle         = shuffle,
            collate_fn      = self.collate_fn
        )

def get_mel_frames(wav, audio_to_mel, min_pad_coverage=0.75, overlap=0.5, order = 'FM', **kwargs):
    '''
    Chops the mel spectograms.

    Order (the shape of the input):
        FM (Frames, Mels) 
        MF (Mels, Frames)
    '''

    if isinstance(wav, str):
        wav, _ = librosa.load(wav)
    wave_slices, mel_slices = compute_partial_slices(len(wav), min_pad_coverage=min_pad_coverage, overlap=overlap, **kwargs)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
    frames = torch.from_numpy(audio_to_mel(wav))
    if order == 'FM':
        frames_batch = [frames[s] for s in mel_slices]
    elif order == 'MF':
        frames_batch = [frames[:,s] for s in mel_slices]
    return frames_batch