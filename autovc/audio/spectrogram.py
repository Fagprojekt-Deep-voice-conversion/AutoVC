"""
Some useful tools for spectrograms. 
"""

import librosa
import numpy as np
# from autovc.utils.hparams import hparams_autoVC as hp
from autovc.utils.hparams import WaveRNNParams 
from scipy.ndimage.morphology import binary_dilation
from autovc.utils.hparams import SpeakerEncoderParams
from pathlib import Path
from typing import Optional, Union
import numpy as np
import webrtcvad
import librosa
import struct
import soundfile as sf
import os
import noisereduce as nr
from autovc.utils.core import retrieve_file_paths
import math, torch
from autovc.speaker_encoder.utils import compute_partial_slices

se_params = SpeakerEncoderParams()
vocoder_params = WaveRNNParams()

def normalize_spec(spectrogram, min_level_db = -100):
    """
    Normalizes a spectrogram.

    Parameters
    ----------
    spectrogram:
        A numpy array containg spectrogram data
    min_level_db:
        The minimum db level to normalize after

    Return
    ------
    spectrogram:
        A numpy array with a normalized spectrogram
    """
    spectrogram = np.clip((spectrogram - min_level_db) / -min_level_db, 0, 1)
    return spectrogram

def denormalize_spec(spectrogram, min_level_db = -100):
    """
    Denormalizes a spectrogram.

    Parameters
    ----------
    spectrogram:
        A numpy array containg spectrogram data
    min_level_db:
        The minimum db level to denormalize after
    
    Return
    ------
    spectrogram:
        A numpy array with a denormalized spectrogram
    """
    spectrogram = (np.clip(spectrogram, 0, 1) * -min_level_db) + min_level_db
    return spectrogram

def amp_to_db(amplitude):
    """Converts an amplitude to decibel"""
    return 20 * np.log10(np.maximum(1e-5, amplitude))

def db_to_amp(power):
    """Converts a power (in decibel) to an amplitude"""
    return np.power(10.0, power * 0.05)

def mel_spec_auto_encoder(wav, sr = 22050, n_mels = 80, n_fft = 2048, hop_length = 275, window_length = 1100, fmin = 40, cut = False, **kwargs):
    """
    Computes a mel spectrogram in the format needed for the Auto Encoder
    The given parameters should match the parameters given to the used vocoder for the best performance.

    Parameters
    ----------
    wav:
        A numpy array with audio content
    sr:
        The sampling rate of the audio content
    n_mels:
        Number of mels to use. See librosa for more info.
        Should match the Auto Encoders feat_dims.
    n_fft:
        The lenght of the window - how many samples to include in Fourier Transformation. See librosa for more info.
    hop_length:
        Number of audio samples between adjacent STFT columns - how far the window moves for each FT. See librosa for more info.
    window_length:
        Each frame of audio is windowed by window of length win_length and then padded with zeros to match n_fft.
    fmin:
        The minimum frequency. See librosa.filters.mel for details.
    cut:
        If true `compute_partial_slices()` is used to cut the mel spectrogram in slices
    **kwargs:
        kwargs are passed to `compute_partial_slices()`

    Return
    ------
    mel_spec:
        A numpy array with the mel spectrogram, this will be a list [(Mels, Frames)] of mel spectrograms if 'cut' is True
    """
    # Short-Time Fourier Transform
    spectrogram = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=window_length)

    # Convert to mel spectrum
    mel_spec = librosa.feature.melspectrogram(S = np.abs(spectrogram), sr = sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin )

    # Convert amplitude to dB
    mel_spec = amp_to_db(mel_spec)

    # Normalised spectrogram    
    mel_spec = normalize_spec(mel_spec)

    # torch
    mel_spec = torch.from_numpy(mel_spec)

    # split into multiple if batch is True
    if cut:
        wave_slices, mel_slices = compute_partial_slices(len(wav),
                                                     min_pad_coverage   = kwargs.pop("min_pad_coverage", 0.75),
                                                     overlap            = kwargs.pop("overlap", 0.5),
                                                     sr                 = kwargs.pop("sr", 22050), # sr from vocoder
                                                     mel_window_step    = kwargs.pop("mel_window_step", 12.5), # from vocoder
                                                     partial_utterance_n_frames = kwargs.pop("partial_utterance_n_frames", 250), 
                                                     **kwargs)
         # Pad last audio frame
        max_wave_length = wave_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
        
        order = "MF"
        if order == 'FM':
            mel_spec = [mel_spec[s] for s in mel_slices]
        elif order == 'MF':
            mel_spec = [mel_spec[:,s] for s in mel_slices]

    return mel_spec

def mel_spec_speaker_encoder(wav, sr = 16000, n_mels = 40, window_length = 25, window_step = 10, cut = False, **kwargs):
    """
    Computes a mel spectrogram in the format needed for the Speaker Encoder

    Parameters
    ----------
    wav:
        A numpy array with audio content
    sr:
        The sampling rate of the audio content
    n_mels:
        Number of mels to use. See librosa for more info.
    window_length:
        In ms. Each frame of audio is windowed by window of length win_length and then padded with zeros to match n_fft.
    window_step:
        In ms. Used to calculate hop length
    cut:
        If true `compute_partial_slices()` is used to cut the mel spectrogram in slices
    **kwargs:
        kwargs are passed to `compute_partial_slices()`
    Return
    ------
    mel_spec:
        A numpy array with the mel spectrogram, this will be a list [(Frames, Mels)] of mel spectrograms if 'cut' is True
    """
    mel_spec = librosa.feature.melspectrogram(
        wav, sr,
        n_fft=int(sr * window_length / 1000),
        hop_length=int(sr * window_step / 1000),
        n_mels=n_mels
    )

    # change type
    mel_spec = mel_spec.astype(np.float32).T

    if cut:
        wave_slices, mel_slices = compute_partial_slices(len(wav),
                                                     min_pad_coverage   = kwargs.pop("min_pad_coverage", 0.75),
                                                     overlap            = kwargs.pop("overlap", 0.5),
                                                     **kwargs)
         # Pad last audio frame
        max_wave_length = wave_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
        
        order = "FM"
        if order == 'FM':
            mel_spec = [mel_spec[s] for s in mel_slices]
        elif order == 'MF':
            mel_spec = [mel_spec[:,s] for s in mel_slices]

    return mel_spec

def mel_spectrogram(wav, model, **kwargs):
    """
    Function for getting a mel spectrogram

    Parameters
    ----------
    wav:
        A numpy array with audio content
    model:
        Chooses which spectrogram function to use. Must be one of 'auto_encoder' or 'speaker_encoder'
    **kwargs
        kwargs are parsed to either `mel_spec_auto_encoder()` or `mel_spec_speaker_encoder()` based on which model is chosen.
    
    Return
    ------
    mel_spec:
        A numpy array with the mel spectrogram 
    """
    if model == "auto_encoder":
        mel_spec = mel_spec_auto_encoder(wav, **kwargs)
    elif model == "speaker_encoder":
        mel_spec = mel_spec_speaker_encoder(wav, **kwargs)
    else:
        raise ValueError(f"'{model}' is not a valid model type. Must be either 'auto_encoder' or 'speaker_encoder'")

    return mel_spec

def compute_partial_slices(n_samples, sr, partial_utterance_n_frames = 160,
                           min_pad_coverage = 0.75,
                           overlap = 0.5,
                           mel_window_step = 10):
    """
    Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain 
    partial utterances of <partial_utterance_n_frames> each. Both the waveform and the mel 
    spectrogram slices are returned, so as to make each partial utterance waveform correspond to 
    its spectrogram. 
    This function has only been tested for mel spectrograms created for the Speaker Encoder.
    
    The returned ranges may be indexing further than the length of the waveform. It is 
    recommended that you pad the waveform with zeros up to wave_slices[-1].stop.
    
    Parameters
    ----------
    n_samples: 
        The number of samples in the waveform
    partial_utterance_n_frames: 
        The number of mel spectrogram frames in each partial utterance (x*10 ms)
    min_pad_coverage: 
        When reaching the last partial utterance, it may or may not have 
        enough frames. If at least <min_pad_coverage> of <partial_utterance_n_frames> are present, 
        then the last partial utterance will be considered, as if we padded the audio. 
        Otherwise, it will be discarded, as if we trimmed the audio. If there aren't enough frames for 1 partial 
        utterance, this parameter is ignored so that the function always returns at least 1 slice.
    overlap: 
        By how much the partial utterance should overlap. If set to 0, the partial utterances are entirely disjoint. 
    mel_window_step:
        How large each frame should be (in ms).
    Return
    ------
    wav_slices:
        The waveform slices as lists of array slices. Index the waveform with these slices to obtain the partial utterances.
    mel_slices:
        Mel spectrogram slices as lists of array slices. Index the mel spectrogram with these slices to obtain the partial utterances.
    """
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1
    
    samples_per_frame = int((sr * mel_window_step / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

    # Compute the slices
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partial_utterance_n_frames])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))
        
    # Evaluate whether extra padding is warranted or not
    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < min_pad_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]
    
    return wav_slices, mel_slices

