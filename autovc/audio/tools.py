"""
Some useful tools to use on audio files. 
To use the tools load an audio file with librosa.load() or the Audio class provided (`from autovc import Audio`)
"""

import inspect
import librosa
import numpy as np
from scipy.ndimage.morphology import binary_dilation
import numpy as np
import webrtcvad
import librosa
import struct
import soundfile as sf
import os
import noisereduce as nr
import math
import shutil

from autovc.utils import retrieve_file_paths

INT_16_MAX = (2 ** 15) - 1


def create_silence_mask(
    wav, 
    sr, 
    vad_window_length = 20, 
    vad_moving_average_width = 8, 
    vad_max_silence_length = 2
):
    """
    Creates a mask where silence frames are set to False. VAD is short for voice activation detection.

    OBS: the mask does not necesarily have same dimensions as input wav, to apply mask use the returned wav.

    Parameters
    ----------
    Default values are based on Speaker Encoder params and might have to be adjusted for other use cases.

    wav:
        A numpy array with audio content
    sr:
        The sampling rate of the audio content
    vad_window_length:
        Window size of the VAD (in milliseconds). Must be either 10, 20 or 30 milliseconds. This sets the granularity of the VAD.
    vad_moving_average_width:
        Number of frames to average together when performing the moving average smoothing. The larger this value, the larger the VAD variations must be to not get smoothed out.
    vad_max_silence_length:
        Maximum number of consecutive silent frames a segment can have.

    Return
    ------
    wav:
        The inputted wav in same dimensions as audio_mask. The end of the origal wav is trimmed to make sure wav is a multiple of the window size. 
    audio_mask:
        A boolean mask where False indicates silence
    """
    # assertions
    assert vad_window_length in [10, 20, 30]
    assert sr in [8000, 16000, 32000, 48000], "The WebRTC VAD only accepts 16-bit mono PCM audio, sampled at 8000, 16000, 32000 or 48000 Hz."

    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sr) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * INT_16_MAX)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sr))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav, audio_mask

def trim_long_silences(wav, sr, **kwargs):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold by applying a silence mask created with `create_silence_mask()`.

    Parameters
    ----------
    wav:
        A numpy array with audio content
    sr:
        The sampling rate of the audio content
    **kwargs
        kwargs are given to `create_silence_mask()`

    Return
    ------
    wav:
        The trimmed wav
    """
    wav, audio_mask = create_silence_mask(wav, sr, **kwargs)
    wav = wav[audio_mask == True]
    return wav

def split_audio(
    wav, 
    sr, 
    save_name = None, 
    save_dir = "data/splitted_wavs/", 
    allowed_pause = 2, 
    remove_silence = False, 
    max_len = 10, 
    fixed_length = None,
    **kwargs):
    """
    Splits the content of the wav into multiple files, based on when there is a longer period if silence.
    Silence is determined with `create_silence_mask()`.
    Files will be saved with a number indicating the order the content appeared in.

    Parameters
    ----------
    wav:
        A numpy array with audio content
    sr:
        The sampling rate of the audio content
    save_name:
        Basename of the resulting files, this is extended with a number indicating the order the content appeared in. 
        If None, the files will NOT be saved. 
        Any leading directories will be ignored and should be given to save_dir instead. (` os.path.split()` makes this easy)
    save_dir:
        Path to save the splitted audio files in. Dir is created if it does not exist.
    allowed_pause:
        Seconds of silence to allow without splitting.
    remove_silence:
        If True the found silence is removed from the splitted audio files.
    max_length:
        The maximum length (in seconds) of a new audio file. The audio file can however exceed 10 seconds if no pauses were found.
    fixed_length:
        if fixed length is different from None a jhard cut of the audio file is performed with an interval of 'fixed_length' seconds, meaning no silence detection is performed
    **kwargs
        kwargs are given to `create_silence_mask()`
    
    Return
    ------
    wavs:
        List of numpy arrays with splitted audio content. The content is not ensured to have equal length.
        The total length will not match that of the original audio file, as `create_silence_mask()` trims the end of the file.

    """

    if fixed_length is not None:
        n_frames = fixed_length*sr
        total_frames = len(wav)
        split_masks = np.array([np.arange(i, i+n_frames) for i in range(0, total_frames, n_frames) if i+n_frames < total_frames])
        # print(splits.shape)
        # return None
        
    else:
        # get slince mask
        wav, audio_mask = create_silence_mask(wav, sr, **kwargs)

        # function for finding consecutive values without silence
        def consecutive(data, stepsize=1):
            return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

        # split files
        allowed_pause = allowed_pause*sr # get pause in frames
        for i, split in enumerate(consecutive(np.where(audio_mask)[0])):
            if i == 0:
                split_masks = [split]
            else:
                # join last subset with new split if difference is less than allowed pause
                new_len = (len(split) + len(split_masks[-1]))/sr
                if (split[-1] - split_masks[-1][-1] <= allowed_pause) and new_len <= max_len:
                    prev = split_masks.pop()
                    if remove_silence:
                        split_masks.append(np.concatenate([prev, split]))
                    else:
                        split_masks.append(np.concatenate([prev,np.arange(prev[-1]+1, split[0]), split]))
                else:
                    split_masks.append(split)
    
    # save splitted files
    if save_name is not None:
        filename = os.path.split(save_name)[-1]
        filename += "" if filename.endswith(".wav") else ".wav"
        os.makedirs(save_dir, exist_ok=True)
    
    wavs = []
    for i, split in enumerate(split_masks):
        wavs.append(wav[split])
        if filename is not None:
            fname = filename.replace(".wav", f"_{str(i+1).zfill(1 + int(math.log10(len(split_masks))))}.wav")
            sf.write(f"{save_dir.strip('/')}/{fname}", wavs[-1], samplerate = sr)


    return wavs

def combine_audio(audio_file_paths, excluded_audio_file_paths = [], sr = 16000, save_name = None):
    """
    Combines multiple audio files into one. 
    Audio fiules are resampled to the given sample rate.

    Parameters
    ----------
    audio_file_paths:
        Path to dir or list of dirs/files to combine. Can also be a list of numpy arrays with audio data.
    excluded_audio_file_paths:
        Path or list of paths to ignore in the given audio_file_paths. Ignored if audio_file_paths is a list of numpy arrays.
    sr:
        The sample rate to use. If audio_file_paths is a list of arrays, this must match their original sample rate
    save_name:
        The filename to use for saving the combined file. If None, no file is saved.

    Return
    ------
    wav_combined:  
        A numpy array with the combined audio data
    """

    # TODO - Combine in order

    is_array = isinstance(audio_file_paths[0], np.ndarray)

    if not is_array:
        audio_file_paths = retrieve_file_paths(audio_file_paths, excluded=excluded_audio_file_paths)

    # combine files
    wav_combined = []
    for wav in audio_file_paths:
        if not is_array:
            wav, _ = librosa.load(wav, sr=sr)
        wav_combined.extend(wav)

    # save file
    if save_name is not None:
        save_name += "" if save_name.endswith(".wav") else ".wav"
        sf.write(save_name, np.array(wav_combined), samplerate = int(np.mean(sr)))
    
    return np.array(wav_combined)

def normalize_volume(wav, target_dBFS = -30, increase_only=False, decrease_only=False):
    """
    Normalizes the volume in audio data.

    Parameters
    ----------
    wav:
        A numpy array with audio content
    target_dBFS:
        The target volume.
    increase_only:
        The volume is only increased, thus if the original volume is higher than the targetted, nothing happens.
    decrease_only:
        The volume is only decreased, thus if the original volume is lower than the targetted, nothing happens.

    Return
    ------
    wav:
        A numpy array containing audio data with a normalized volume.
    """
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))

def remove_noise(wav, sr, **kwargs):
    """
    Reduces the noise in audio data.

    Parameters
    ----------
    wav:
        A numpy array with audio content
    sr:
        The sampling rate of the audio content
    **kwargs:
        kwargs are given to `noisereduce.reduce_noise()` 
        see: [noisereduce pypi](https://pypi.org/project/noisereduce/)
    
    Return
    ------
    wav:
        A numpy array containing audio data with reduced noise
    """
    return nr.reduce_noise(y=wav, sr=sr, **kwargs)

def rename_files(dir_path, new_dir_path, new_file_name):
    files = os.listdir(dir_path)
    files.sort()
    os.makedirs(new_dir_path, exist_ok=True)

    for i, file in enumerate(files):
        save_name = os.path.join(new_dir_path, new_file_name)
        save_name += "" if save_name.endswith(".wav") else ".wav"
        fname = save_name.replace(".wav", f"_{str(i+1).zfill(1 + int(math.log10(len(files))))}.wav")
        print(os.path.join(dir_path, file), fname)
        shutil.copy(os.path.join(dir_path, file), fname)

    # print(files)


# add annotations to use for preprocessing
trim_long_silences.__allowed_args__ = inspect.getfullargspec(trim_long_silences).args
trim_long_silences.__allowed_kw__ = inspect.getfullargspec(create_silence_mask).args

normalize_volume.__allowed_args__ = inspect.getfullargspec(normalize_volume).args
normalize_volume.__allowed_kw__ = []

remove_noise.__allowed_args__ = inspect.getfullargspec(remove_noise).args
remove_noise.__allowed_kw__ = inspect.getfullargspec(nr.reduce_noise).args

if __name__ == "__main__":
    # wav, sr = librosa.load("data/SMK_train/Hilde.wav", sr = 16000)
    # wavs = split_audio(wav, sr, save_name="hilde.wav", allowed_pause = 2, max_len = 6, save_dir = "data/newest_trial/hilde")

    # wav, sr = librosa.load("data/SMK_HY_long.wav", sr = 16000)
    # wavs = split_audio(wav, sr, save_name="louise.wav", allowed_pause = 2, max_len = 6, save_dir = "data/newest_trial/louise")

    # wav, sr = librosa.load("data/yang_long.wav", sr = 16000)
    # wavs = split_audio(wav, sr, save_name="yang.wav", allowed_pause = 2, max_len = 6, save_dir = "data/newest_trial/yang")
    # remove_noise(wav, sr)

    # combine_audio("data/newest_trial/hilde_subset", save_name = "data/hilde_subset.wav", sr = 16000)

    # wav, sr = librosa.load("AutoVC/data/SMK_train/Hilde.wav", sr = 16000)
    # wavs = split_audio(wav, sr, save_name="hilde.wav", fixed_length=10, save_dir = "Deep_voice_conversion/data/SMK_original/hilde")

    # rename_files("AutoVC/data/SMK_train/hyang_smk", "Deep_voice_conversion/data/SMK_original/yangSMK", "yangSMK.wav")

    rename_files("../AutoVC/data/HY", "data/SMK_original/louise", "louise.wav")
