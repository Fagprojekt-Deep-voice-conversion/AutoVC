#ALOT OF CODE HAS BEEN OBTAINED FROM THIS ARTICLE: https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0

import librosa
import numpy as np
# from autovc.utils.hparams import hparams_autoVC as hp
from autovc.utils.hparams_NEW import WaveRNNParams as hparams


def normalize(S):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)

def denormalize(S):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db

def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def audio_to_melspectrogram(waveform):
    '''
    Loads a .wav file and converts audio to mel spectrogram
    params:
        - path (str): path to wav fil
    returns:
        - Mel spectrogram        
    '''


    # Load .wav to audio waveform
    if isinstance(waveform, str):
        waveform, _ = librosa.load(waveform, sr=hparams.sample_rate)

    # Short-Time Fourier Transform
    spectrogram = librosa.stft( waveform,                   # Audio Time Series as input
                                n_fft=hparams.n_fft,            # The lenght of the window - how many samples to include in Fourier Transformation
                                hop_length=hparams.hop_length,  # The number of samples between adjacent STFT columns - how far the window moves for each FT
                                win_length=hparams.win_length,  # The lenght of the window - pad with zeros to match n_fft
                                )

    # Convert to mel spectrum
    melspectrogram = librosa.feature.melspectrogram(S = np.abs(spectrogram),
                                                    sr = hparams.sample_rate,
                                                    n_fft=hparams.n_fft,
                                                    n_mels=hparams.num_mels,    # The number of mels
                                                    fmin=hparams.fmin           # The minimum frequency
                                                    )

    # Convert amplitude to dB
    melspectrogram = amp_to_db(melspectrogram)
    

    # Return normalised spectrogram    
    return normalize(melspectrogram)



    

       




