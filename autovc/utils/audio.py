#ALOT OF CODE HAS BEEN OBTAINED FROM THIS ARTICLE: https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0

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


######### FROM VOCODER #########
vocoder_params = WaveRNNParams()
def normalize(S, min_level_db = None):
    min_level_db = vocoder_params.min_level_db if min_level_db is None else min_level_db
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def denormalize(S, min_level_db = None):
    min_level_db = vocoder_params.min_level_db if min_level_db is None else min_level_db
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

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
        waveform, _ = librosa.load(waveform, sr=vocoder_params.sample_rate)

    # Short-Time Fourier Transform
    spectrogram = librosa.stft( waveform,                   # Audio Time Series as input
                                n_fft=vocoder_params.n_fft,            # The lenght of the window - how many samples to include in Fourier Transformation
                                hop_length=vocoder_params.hop_length,  # The number of samples between adjacent STFT columns - how far the window moves for each FT
                                win_length=vocoder_params.win_length,  # The lenght of the window - pad with zeros to match n_fft
                                )

    # Convert to mel spectrum
    melspectrogram = librosa.feature.melspectrogram(S = np.abs(spectrogram),
                                                    sr = vocoder_params.sample_rate,
                                                    n_fft=vocoder_params.n_fft,
                                                    n_mels=vocoder_params.feat_dims,    # The number of mels
                                                    fmin=vocoder_params.fmin           # The minimum frequency
                                                    )

    # Convert amplitude to dB
    melspectrogram = amp_to_db(melspectrogram)

    # Return normalised spectrogram    
    return normalize(melspectrogram)


######### FROM SPEAKER ENCODER #########
int16_max = (2 ** 15) - 1
se_params = SpeakerEncoderParams()


def preprocess_wav(wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform's sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(wav, str) or isinstance(wav, Path):
        wav, source_sr = librosa.load(wav, sr=None)
    
    # Resample the wav if needed
    if source_sr is not None and source_sr != se_params.sampling_rate:
        wav = librosa.resample(wav, source_sr, se_params.sampling_rate)
        
    # Apply the preprocessing: normalize volume and shorten long silences 
    #wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    wav = trim_long_silences(wav)
    
    return wav



def create_audio_mask(wav):
    """"
    Creates an audio mask, where False indicates silence.

    :param wav: a numpy array with the content of a wav file
    :return audio_mask: the calculated audio mask
    """
    # Compute the voice detection window size
    samples_per_window = (se_params.vad_window_length * se_params.sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=se_params.sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, se_params.vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(se_params.vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav, audio_mask

def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # # Compute the voice detection window size
    # samples_per_window = (hparams.vad_window_length * hparams.sampling_rate) // 1000
    
    # # Trim the end of the audio to have a multiple of the window size
    # wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # # Convert the float waveform to 16-bit mono PCM
    # pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # # Perform voice activation detection
    # voice_flags = []
    # vad = webrtcvad.Vad(mode=3)
    # for window_start in range(0, len(wav), samples_per_window):
    #     window_end = window_start + samples_per_window
    #     voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
    #                                      sample_rate=hparams.sampling_rate))
    # voice_flags = np.array(voice_flags)
    
    # # Smooth the voice detection with a moving average
    # def moving_average(array, width):
    #     array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
    #     ret = np.cumsum(array_padded, dtype=float)
    #     ret[width:] = ret[width:] - ret[:-width]
    #     return ret[width - 1:] / width
    
    # audio_mask = moving_average(voice_flags, hparams.vad_moving_average_width)
    # audio_mask = np.round(audio_mask).astype(np.bool)
    
    # # Dilate the voiced regions
    # audio_mask = binary_dilation(audio_mask, np.ones(hparams.vad_max_silence_length + 1))
    # audio_mask = np.repeat(audio_mask, samples_per_window)
    wav, audio_mask = create_audio_mask(wav)
    
    return wav[audio_mask == True]

# own tools

def split_audio(wav_path, save_folder = None, allowed_pause = 2, remove_silence = False, max_len = 10):
    """
    Chops the content of the wav file into multiple files, based on when there is
    a longer period if silence. Files will be saved with a number indicating the order the content appeared in.

    :param wav: path to wav file
    :param save_folder: folder to save files in
    :param allowed_pause: number seconds of silence to allow in a sound file
    :param remove_silence: whether to remove intermediate silence in each of the new wav files
    :param max_len: the maximum length (in seconds) of a new sound file. The sound file can however exceed 10 seconds if no pauses were found
    :return: content of the wav file seperated in multiple files 
    """
    # load wav
    wav, source_sr = librosa.load(wav_path, sr=None)

    wav, audio_mask = create_audio_mask(wav)

    # function for finding consecutive values without silence
    def consecutive(data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


    allowed_pause = allowed_pause*source_sr
    for i, split in enumerate(consecutive(np.where(audio_mask)[0])):
        if i == 0:
            joined_splits = [split]
        else:
            # join last subset with new split if difference is less than allowed pause
            new_len = (len(split) + len(joined_splits[-1]))/source_sr
            if (split[-1] - joined_splits[-1][-1] <= allowed_pause) and new_len <= max_len:
                prev = joined_splits.pop()
                if remove_silence:
                    joined_splits.append(np.concatenate([prev, split]))
                else:
                    joined_splits.append(np.concatenate([prev,np.arange(prev[-1]+1, split[0]), split]))
            else:
                joined_splits.append(split)
    
    # save chopped files
    filename = os.path.split(wav_path)[-1]
    wav_splitted = []
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        
        for i, split in enumerate(joined_splits):
            fname = filename.replace(".wav", f"_{str(i+1).zfill(3)}.wav")
            wav_splitted.append(wav[split])
            sf.write(f"{save_folder}/{fname}", wav_splitted[-1], samplerate = source_sr)


    return wav_splitted

def remove_noise(wav, sample_rate = None):
    """
    remove noise from audio
    """
    if isinstance(wav, str):
        wav, sample_rate = librosa.load(wav, sr=None)
    
    return nr.reduce_noise(y=wav, sr=sample_rate)

def combine_audio(audio_clip_paths, save_name = "combined.wav"):
    """
    Combines multiple audio files into one. Sample rates must be equal or near equal to avoid alien sounding voices
    """
    if os.path.isdir(audio_clip_paths):
        walk = [w for w in os.walk(audio_clip_paths)][0]
        root, _, data = [w for w in walk]
        audio_clip_paths = [root + "/" + d for d in data]
    
    wav_combined = []
    sr = []   
    for file in audio_clip_paths:
        wav, source_sr = librosa.load(file, sr=None)
        wav_combined.extend(wav)
        sr.append(source_sr)

    sf.write(save_name, np.array(wav_combined), samplerate = int(np.mean(sr)))
    return wav_combined

# def change_audio_format(audio_path, new_format = "wav", save_folder = None):
#     """
#     Changes the audio format. 
#     OBS: requires ffmpeg
#     """

#     # get file info
#     root, file = os.path.split(audio_path)
#     filename, fileext = os.path.splitext(file)

#     # import required modules
#     import subprocess
    
#     # convert mp3 to wav file
#     subprocess.call(['ffmpeg', '-i', f'{audio_path}',
#                     f'{root if save_folder is None else save_folder}/{filename}.{new_format}'])



if __name__ == "__main__":
    split_audio("data/long_hilde/Hilde.wav", "data/long_hilde/chopped")
    # combine_audio("data/samples")

    # change_audio_format("data/samples/chooped7.wav", new_format="mp3")