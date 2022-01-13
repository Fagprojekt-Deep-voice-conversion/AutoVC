from ctypes import ArgumentError
import numpy as np 
import torch
import os
from torch.utils.data import DataLoader, Dataset
from autovc.utils.audio import audio_to_melspectrogram, get_mel_frames
from autovc.speaker_encoder.model import SpeakerEncoder
from autovc.speaker_encoder.utils import *
from torch.nn.functional import pad
from autovc.utils.audio import remove_noise, preprocess_wav
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

    def __init__(self, speaker_encoder, data_path = None, data_path_excluded= None, wav_files = None,  chop = False, **kwargs):
        super(TrainDataLoader, self).__init__()

        # Load wav files. Create spectograms and embeddings
        if wav_files is not None:
            self.wav_files = wav_files
        elif data_path is not None:
            self.wav_files = retrieve_file_paths(data_path, excluded=data_path_excluded if data_path_excluded is not None else [])
        else:
            raise ArgumentError(f"Either data_path or wav_files must be different from None")
                
        self.mel_spectograms, self.embeddings, N = [], [], len(self.wav_files)

        print("Creating mel spectrograms and embeddings...")
        progbar(0, N)
        for i, wav in enumerate(self.wav_files):
            # make nice sounds

            if chop:
                # Chops the mel spectograms in size of 'partial_n_utterances


                wav = preprocess_wav(wav)

                
                mel_frames = get_mel_frames(wav,
                                            audio_to_melspectrogram,
                                            order = 'MF',
                                            sr = vocoder_params.sample_rate, 
                                            mel_window_step = vocoder_params.mel_window_step, 
                                            partial_utterance_n_frames = 250  )
                # Get embeddings of speech
                embeds     = speaker_encoder.embed_utterance(wav)

                # Add to dataset - embeddings are cloned to match the number of mel frames
                self.mel_spectograms.extend(mel_frames)
                self.embeddings.extend([embeds.clone() for _ in range(len(mel_frames))])
            else:
                # Compute mel spectogram and speaker embeddings
                mel_frames = torch.from_numpy(audio_to_melspectrogram(wav))
                embeds     = speaker_encoder.embed_utterance(wav)

                # Add to dataset
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
        return DataLoader(
            self,  
            batch_size=batch_size, 
            num_workers= num_workers, 
            shuffle=shuffle,
            collate_fn = self.collate_fn
        )


class SpeakerEncoderDataLoader(Dataset):
    def __init__(self, data_dict, device = 'cpu'):
        super(SpeakerEncoderDataLoader, self).__init__()
        self.device = device
        # Find wav files in dictionary of data        
        wav_files = [sum([retrieve_file_paths(data_dir_path) for data_dir_path in speaker_data_dir], []) for speaker_data_dir in data_dict.values()]

        # Compute mel spectograms
        speakers = len(data_dict.keys())
        N = sum([len(d) for d in wav_files])
        self.datasets = [[] for _ in wav_files]
        t = 0
        print("Creating mel spectograms ...")
        progbar(t, N)
        for i in range(speakers):
            for wav in wav_files[i]:
                wav = preprocess_wav(wav)
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

        return torch.stack(tuple(torch.stack(b) for b in list(zip(*batch)) )).to(self.device)

    def get_dataloader(self, batch_size=2, shuffle=False,  num_workers=0, pin_memory=False, **kwargs):
        return DataLoader(
            self,  
            batch_size      = batch_size, 
            num_workers     = num_workers, 
            shuffle         = shuffle,
            collate_fn      = self.collate_fn
        )

