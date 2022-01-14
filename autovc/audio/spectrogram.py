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

def compute_partial_slices():
    pass

def split_melspectrograms():
    pass