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

def audio_to_melspectrogram():
    pass

def compute_partial_slices(n_samples, sr, partial_utterance_n_frames = 160,
                           min_pad_coverage = 0.75,
                           overlap = 0.5,
                           mel_window_step = 10):
    """
    Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain 
    partial utterances of <partial_utterance_n_frames> each. Both the waveform and the mel 
    spectrogram slices are returned, so as to make each partial utterance waveform correspond to 
    its spectrogram. 
    This function assumes that the mel spectrogram parameters used are those 
    defined in params_data.py.
    
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

def split_melspectrograms():
    pass