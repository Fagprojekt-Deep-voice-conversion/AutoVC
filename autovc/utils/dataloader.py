import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import pad
from autovc.audio.spectrogram import mel_spec_auto_encoder, mel_spec_speaker_encoder, compute_partial_slices
from autovc.utils import retrieve_file_paths, close_progbar, progbar
from autovc.utils.hparams import SpeakerEncoderParams, AutoEncoderParams
from autovc import audio
import inspect



class AutoEncoderDataset(Dataset):
    '''
    A Dataset wrapper class for training AutoVC.
    Takes a path to a folder with data (.wav files) and makes a generator object (data loader) 
    which batches spectorgrams of utterances and the corresponding speaker identiy embeddings.
    '''

    def __init__(self, 
        speaker_encoder, 
        data_path, 
        data_path_excluded= [], 
        use_mean_speaker_embedding = False, 
        sr = AutoEncoderParams["spectrogram"]["sr"],
        preprocess = SpeakerEncoderParams["dataset"]["preprocess"],
        preprocess_args = SpeakerEncoderParams["dataset"]["preprocess_args"],
        **kwargs
    ):
        """
        Initilialises the dataset class.

        Parameters
        ----------
        speaker_encoder:
            An initialised speaker encoder to create speaker embeddings with
        data_path:
            Path to the data. See `utils.retrieve_file_paths()` for input format.
        data_path_excluded:
            Paths to exclude from the data. Same as excluded in  `utils.retrieve_file_paths()`.
        use_mean_speaker_embedding:
            If true, the name of the wav file will be compared to the speaker names found in speaker_encoder.speakers.keys(), 
            if the speaker name is found in the file name, the matching mean speaker embedding will be used.
            To use this proparly, the speaker_encoder should learn the necesary mean speaker embedding before creating the data set.
        sr:
            Sample rate to load the data with
        **kwargs:
            kwargs are passed to `audio.spectrogram.mel_spec_auto_encoder()`.
            The cut parameter defaults to True.
        """
        super(AutoEncoderDataset, self).__init__()

        # Load wav files
        self.wav_files = retrieve_file_paths(data_path, excluded=data_path_excluded)

        # initial values        
        self.mel_spectograms, self.embeddings, N = [], [], len(self.wav_files)

        # pop cut kwarg
        cut = kwargs.pop("cut", True)

        # create data set
        print("Creating mel spectrograms and embeddings...")
        progbar(0, N)
        for i, wav in enumerate(self.wav_files):
            data = audio.Audio(wav, sr = sr)

            # TODO - preprocess
            data.preprocess(*preprocess, **preprocess_args)

            # get mel spectrogram
            mel_frames = audio.spectrogram.mel_spec_auto_encoder(data.wav, sr = data.sr, cut = cut, **kwargs)

            # Get embeddings of speech
            found_speaker = False
            if use_mean_speaker_embedding:
                for speaker in speaker_encoder.speakers.keys():
                    if speaker in wav:
                        embeds = speaker_encoder.speakers[speaker]
                        found_speaker = True
            if not found_speaker:
                # name = 'hilde' if 'hilde' in wav else 'HaegueYang'
                # embeds = speaker_encoder.speakers[name]
                embeds     = speaker_encoder.embed_utterance(data.wav)
            
            # append to data set
            if cut:
                self.mel_spectograms.extend(mel_frames)
                self.embeddings.extend([embeds.clone() for _ in range(len(mel_frames))])
            else:
                self.mel_spectograms.append(mel_frames)
                self.embeddings.append(embeds)

            progbar(i+1, N)
        close_progbar()

    def __len__(self):
        return len(self.mel_spectograms)

    def __getitem__(self, index):
        return self.mel_spectograms[index], self.embeddings[index]


    def collate_fn(self, batch):
        '''
        Allow unequal length wavs to be batched together
        If spectograms in batch are of unequal size the smaller are padded with zeros to match the size of the largest.
        '''
        spectrograms, embeddings = zip(*batch)
        max_length = max([spectogram.size(-1) for spectogram in spectrograms]) 
        padded_spectrograms = [pad(spectogram, (0, max_length - spectogram.size(-1))) for spectogram in spectrograms]

        return torch.stack(padded_spectrograms), torch.stack(embeddings)

    def get_dataloader(self, batch_size=1, shuffle=False, **kwargs):
        """
        Creates a data loader from the data set

        Parameters
        ----------
        batch_size:
            Batch size to use
        shuffle:
            bool telling whether or not to shuffle the data
        **kwargs:
            kwargs are given to `torch.DataLoader()` (see [documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader))
        """
        
        data_loader = DataLoader(
            self,  
            batch_size=batch_size,  
            shuffle=shuffle,
            collate_fn = self.collate_fn,
            **kwargs
        )
        return data_loader


class SpeakerEncoderDataset(Dataset):
    def __init__(self, 
        data_path, 
        data_path_excluded = [],
        sr = SpeakerEncoderParams["spectrogram"]["sr"],
        device = 'cpu', 
        preprocess = SpeakerEncoderParams["dataset"]["preprocess"],
        preprocess_args = SpeakerEncoderParams["dataset"]["preprocess_args"],
        **kwargs
    ):
        """
        Initilialises the dataset class.

        Parameters
        ----------
        data_path:
            dictionary with speaker name as key and data directory or list of data as key.
        data_path_excluded:
            List of files to exclude from the data loader
        sr:
            Sample rate to load the data with
        device:
            Torch device to store data in.
        **kwargs:
            kwargs are passed to `audio.spectrogram.mel_spec_speaker_encoder()`.
            The cut parameter defaults to True.
        """
        super(SpeakerEncoderDataset, self).__init__()

        # set values
        self.device = device
        cut = kwargs.pop("cut", True)
        

        # Find wav files in dictionary of data        
        wav_files = [sum([retrieve_file_paths(data_dir_path, excluded=data_path_excluded) for data_dir_path in speaker_data_dir], []) for speaker_data_dir in data_path.values()]

        # initial values
        N = sum([len(d) for d in wav_files])
        speakers = len(data_path.keys())
        self.datasets = [[] for _ in wav_files]

        # create data set
        print("Creating mel spectograms ...")
        progbar(0, N)
        for i in range(speakers):
            for wav in wav_files[i]:
                data = audio.Audio(wav, sr = sr)
                
                # TODO - preprocess
                data.preprocess(*preprocess, **preprocess_args)

                # get mel spectrograms
                frames_batch = audio.spectrogram.mel_spec_speaker_encoder(data.wav, sr = data.sr, cut = cut, **kwargs)
                
                # append to data set
                if cut:
                    self.datasets[i].extend(frames_batch)
                else:
                    self.datasets[i].append(frames_batch)
                
                
                progbar(i+1, N)
        
        print(f"The datasets are of lengths: {[len(d) for d in self.datasets]}")

    def __getitem__(self, i):
        return tuple(d[i % len(d)] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)

    def collate_fn(self, batch):
        return torch.stack(tuple(torch.stack(b) for b in list(zip(*batch)) )).to(self.device)

    def get_dataloader(self, batch_size=1, shuffle=False, **kwargs):
        """
        Creates a data loader from the data set

        Parameters
        ----------
        batch_size:
            Batch size to use
        shuffle:
            bool telling whether or not to shuffle the data
        **kwargs:
            kwargs are given to `torch.DataLoader()` (see [documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader))
        """
        
        data_loader = DataLoader(
            self,  
            batch_size=batch_size,  
            shuffle=shuffle,
            collate_fn = self.collate_fn,
            **kwargs
        )
        return data_loader

# add function annotations
AutoEncoderDataset.__allowed_args__ = inspect.getfullargspec(AutoEncoderDataset.__init__).args
AutoEncoderDataset.__allowed_kw__ = inspect.getfullargspec(mel_spec_auto_encoder).args + inspect.getfullargspec(compute_partial_slices).args

AutoEncoderDataset.get_dataloader.__allowed_args__ = inspect.getfullargspec(AutoEncoderDataset.get_dataloader.__init__).args
AutoEncoderDataset.get_dataloader.__allowed_kw__ = inspect.getfullargspec(DataLoader.__init__).args

SpeakerEncoderDataset.__allowed_args__ = inspect.getfullargspec(SpeakerEncoderDataset.__init__).args
SpeakerEncoderDataset.__allowed_kw__ = inspect.getfullargspec(mel_spec_speaker_encoder).args + inspect.getfullargspec(compute_partial_slices).args

SpeakerEncoderDataset.get_dataloader.__allowed_args__ = inspect.getfullargspec(SpeakerEncoderDataset.get_dataloader.__init__).args
SpeakerEncoderDataset.get_dataloader.__allowed_kw__ = inspect.getfullargspec(DataLoader.__init__).args