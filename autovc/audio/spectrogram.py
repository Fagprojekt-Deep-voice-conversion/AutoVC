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
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def denormalize_spec():
    pass

def amp_to_db():
    pass

def db_to_amp():
    pass

def audio_to_melspectrogram():
    pass

def compute_partial_slices():
    pass

def split_melspectrograms():
    pass