import numpy as np 
import torch
import os
from torch.utils.data import DataLoader, Dataset
from autovc.utils.audio import audio_to_melspectrogram
from autovc.speaker_encoder.model import SpeakerEncoder
from autovc.speaker_encoder.utils import *
from torch.nn.functional import pad
from autovc.utils.audio import remove_noise
from autovc.utils.progbar import progbar
from autovc.utils.core import retrieve_file_paths

class TrainDataLoader(Dataset):
    '''
    A Data Loader class for training AutoVC.
    Takes a path to a folder with data (.wav files) and makes a generator object (data loader) 
    which batches spectorgrams of utterances and the corresponding speaker identiy embeddings.
    
    If spectograms in batch are of unequal size the smaller are padded with zeros to match the size of the largest.
    '''

    def __init__(self, data_dir_path, speaker_encoder):
        super(TrainDataLoader, self).__init__()

        # Load wav files. Create spectograms and embeddings
        # self.wav_files = [os.path.join(dirpath, filename) for dirpath , _, directory in os.walk(data_dir_path) for filename in directory]
        self.wav_files = retrieve_file_paths(data_dir_path)
        
        print(self.wav_files)
        # self.mel_spectograms = [torch.from_numpy(audio_to_melspectrogram(wav)) for wav in self.wav_files]
        # self.embeddings      = [speaker_encoder.embed_utterance(wav) for wav in self.wav_files]
        self.mel_spectograms, self.embeddings, N = [], [], len(self.wav_files)

        print("Creating mel spectrograms and embeddings...")
        progbar(0, N)
        for i, wav in enumerate(self.wav_files):
            # make nice sounds
            self.mel_spectograms.append(torch.from_numpy(audio_to_melspectrogram(wav)))
            self.embeddings.append(speaker_encoder.embed_utterance(wav))
            progbar(i+1, N)

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
        padded_spectrograms = [pad(spectogram, (1, max_length - spectogram.size(-1))) for spectogram in spectrograms]

        return torch.stack(padded_spectrograms), torch.stack(embeddings)

    def get_dataloader(self, batch_size=1, shuffle=False,  num_workers=0, pin_memory=False, **kwargs):
        return torch.utils.data.DataLoader(
            self,  
            batch_size=batch_size, 
            num_workers= num_workers, 
            shuffle=shuffle,
            collate_fn = self.collate_fn
        )


class SpeakerEncoderDataLoader(Dataset):
    def __init__(self, data_dict):
        super().__init__()

        # Find wav files in dictionary of data        
        wav_files = [sum([retrieve_file_paths(data_dir_path)[:200] for data_dir_path in speaker_data_dir], []) for speaker_data_dir in data_dict.values()]

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

def get_mel_frames(wav, audio_to_mel, min_pad_coverage=0.75, overlap=0.5):
    wav, _ = librosa.load(wav)
    wave_slices, mel_slices = compute_partial_slices(len(wav), min_pad_coverage=min_pad_coverage, overlap=overlap)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
    frames = torch.from_numpy(audio_to_mel(wav))
    frames_batch = [frames[s] for s in mel_slices]
    return frames_batch