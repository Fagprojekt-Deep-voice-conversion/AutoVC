# List of old audio functions

Only files in the autovc folder are included

| Function          | Origin script         | Used in       | Short description         | Category      | 
|-------------------|-----------------------|---------------|---------------------------|---------------|
| `normalize()`     | utils/audio.py        | `audio_to_melspectrogram()` | Normalizes spectrogram| Spectrogram |
| `denormalize()`   | utils/audio.py | Nowhere       | Denormalizes spectrogram| Spectrogram|
| `amp_to_db()`     | utils/audio.py | `audio_to_melspectrogram()`| Converts amplitude to decibel| Spectrogram|
| `db_to_amp()`     | utils/audio.py | Nowhere | Converts amplitude to decibel| Spectrogram |
| `audio_to_melspectrogram()` | utils/audio.py | `VoiceConverter.convert()`<br>`TrainDataLoader()`| Create mel spec | Spectrogram |
| `preprocess_wav()`| utils/audio.py        |`VoiceConverter.convert()`<br>`TrainDataLoader()`<br>`SpeakerEncoderDataLoader()` | Use different audio tools on audio file | sound tool |
| `create_audio_mask()`     | utils/audio.py        |`split_audio()`<br>`trim_long_silences()`| Creates a mask indicating silence | sound tool |
| `trim_long_silences()`     | utils/audio.py        |`preprocess_wav()` | Remove long periods with silence | sound tool |
| `normalize_volume()`     | utils/audio.py        | `preprocess_wav()`| Set volume of sound| sound tool |
| `split_audio()`     | utils/audio.py        | Used before everything | Split audio in multiple sound clips | sound tool|
| `remove_noise()`     | utils/audio.py        | `VoiceConverter.convert()` <br> `preprocess_wav()`| Removes noise from sound | sound tool |
| `combine_audio()`     | utils/audio.py        | Nowhere | Combines audio files into 1 | sound tool |
| `get_mel_frames()`     | utils/audio.py        | `TrainDataLoader()`<br>`SpeakerEncoderDataLoader()`| Creates a batch of mel specs | spectrogram |
| `wav_to_mel_spectrogram()`     | speaker_encoder/utils.py        | `se.embed_utterance()`<br> `SpeakerEncoderDataLoader()` | Creates a mel spec  | spectrogram |
| `compute_partial_slices()`     | speaker_encoder/utils.py        | `se.embed_utterance()`<br>`utils/audio.get_mel_frames()`| splits mel specs and waveforms | spectrogram |


## Old function definitions

### audio.py

```python
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
from autovc.utils.core import retrieve_file_paths
import math, torch
from autovc.speaker_encoder.utils import compute_partial_slices

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
                   source_sr: Optional[int] = None,
                   trim = False):
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
        wav, source_sr = librosa.load(wav, sr=vocoder_params.sample_rate)
    
    # Resample the wav if needed
    # if source_sr is not None and source_sr != se_params.sampling_rate:
    #     wav = librosa.resample(wav, source_sr, se_params.sampling_rate)
        
    # Apply the preprocessing: normalize volume and shorten long silences 
    #wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    wav = normalize_volume(wav, -20)
    wav = remove_noise(wav, sample_rate=vocoder_params.sample_rate)
    if trim:
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
    wav, audio_mask = create_audio_mask(wav)
    
    return wav[audio_mask == True]

def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))

# own tools

def split_audio(wav_path, save_folder, allowed_pause = 2, remove_silence = False, max_len = 10):
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
    os.makedirs(save_folder, exist_ok=True)
    
    for i, split in enumerate(joined_splits):
        fname = filename.replace(".wav", f"_{str(i+1).zfill(1 + int(math.log10(len(joined_splits))))}.wav")
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

    audio_clip_paths = retrieve_file_paths(audio_clip_paths)
    
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
def get_mel_frames(wav, audio_to_mel, min_pad_coverage=0.75, overlap=0.5, order = 'FM', **kwargs):
    '''
    Chops the mel spectograms.
    params:
        wav: file or waveform (array) of audio
        
        audio_to_mel: function to convert audio to spectogram

        min_pad_coverage:   when chopping into equal sizes the last chop is often larger than the remaining audio. 
                            This parameters ensures that a percentage of the remaining audio is in the chop before just padding

        overlap: How much each consecutive frame overlap

        Order (the shape of the input):
            FM (Frames, Mels) 
            MF (Mels, Frames)
    '''

    if isinstance(wav, str):
        wav, _ = librosa.load(wav, sr = kwargs.get('sr', None))

    wave_slices, mel_slices = compute_partial_slices(len(wav),
                                                     min_pad_coverage   = min_pad_coverage,
                                                     overlap            = overlap,
                                                     **kwargs)

    # Pad last audio frame
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

    # Retrieve mel spectograms in chops
    frames = torch.from_numpy(audio_to_mel(wav))
    if order == 'FM':
        frames_batch = [frames[s] for s in mel_slices]
    elif order == 'MF':
        frames_batch = [frames[:,s] for s in mel_slices]

    for i,s in enumerate(wave_slices[:-1]):
        sf.write(f'data/chopped{i}.wav', np.array(wav[s]), samplerate =  kwargs.get('sr', None))
        
    return frames_batch


if __name__ == "__main__":
    split_audio("data/SMK_train/Hilde.wav", "data/SMK_train/newest_trial/hilde")
    # combine_audio("data/samples")

    # change_audio_format("data/samples/chooped7.wav", new_format="mp3")
```


### speaker_encoder.utils.py

```python
from autovc.utils.hparams import SpeakerEncoderParams as hparams
# from pathlib import __path__
import numpy as np
import librosa
# from autovc.speaker_encoder.model import SpeakerEncoder
# from autovc.speaker_encoder import audio
# from pathlib import Path
import numpy as np
import torch
from pathlib import Path
# int16_max = (2 ** 15) - 1
# _model = None # type: SpeakerEncoder
# _device = None # type: torch.device
hparams = hparams()


# def load_model(weights_fpath: Path, device=None):
#     """
#     Loads the model in memory. If this function is not explicitely called, it will be run on the 
#     first call to embed_frames() with the default weights file.
    
#     :param weights_fpath: the path to saved model weights.
#     :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda"). The 
#     model will be loaded and will run on this device. Outputs will however always be on the cpu. 
#     If None, will default to your GPU if it"s available, otherwise your CPU.
#     """

#     global _model, _device
#     if device is None:
#         _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     elif isinstance(device, str):
#         _device = torch.device(device)
#     # _model = SpeakerEncoder(_device, torch.device("cpu"))
#     _model = SpeakerEncoder(_device)
#     checkpoint = torch.load(weights_fpath,map_location=torch.device('cpu'))
#     _model.load_state_dict(checkpoint["model_state"])
#     _model.eval()
#     print("Loaded encoder \"%s\" trained to step %d" % (weights_fpath, checkpoint["step"]))
    
#     return _model

# def is_loaded():
#     return _model is not None


def wav_to_mel_spectrogram(wav, sampling_rate = None, mel_window_length = None, mel_window_step = None, mel_n_channels = None):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    if isinstance(wav, str) or isinstance(wav, Path):
        wav, source_sr = librosa.load(wav, sr=None)

    # set default values
    sampling_rate = sampling_rate if sampling_rate is not None else hparams.sampling_rate
    mel_window_length = mel_window_length if mel_window_length is not None else hparams.mel_window_length
    mel_window_step = mel_window_step if mel_window_step is not None else hparams.mel_window_step
    mel_n_channels = mel_n_channels if mel_n_channels is not None else hparams.mel_n_channels


    frames = librosa.feature.melspectrogram(
        wav,
        sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )

    return frames.astype(np.float32).T



def compute_partial_slices(n_samples, 
                           partial_utterance_n_frames = hparams.partials_n_frames,
                           min_pad_coverage = 0.75,
                           overlap = 0.5,
                           sr = hparams.sampling_rate,
                           mel_window_step = hparams.mel_window_step ):
    """
    Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain 
    partial utterances of <partial_utterance_n_frames> each. Both the waveform and the mel 
    spectrogram slices are returned, so as to make each partial utterance waveform correspond to 
    its spectrogram. This function assumes that the mel spectrogram parameters used are those 
    defined in params_data.py.
    
    The returned ranges may be indexing further than the length of the waveform. It is 
    recommended that you pad the waveform with zeros up to wave_slices[-1].stop.
    
    :param n_samples: the number of samples in the waveform
    :param partial_utterance_n_frames: the number of mel spectrogram frames in each partial 
    utterance
    :param min_pad_coverage: when reaching the last partial utterance, it may or may not have 
    enough frames. If at least <min_pad_coverage> of <partial_utterance_n_frames> are present, 
    then the last partial utterance will be considered, as if we padded the audio. Otherwise, 
    it will be discarded, as if we trimmed the audio. If there aren't enough frames for 1 partial 
    utterance, this parameter is ignored so that the function always returns at least 1 slice.
    :param overlap: by how much the partial utterance should overlap. If set to 0, the partial 
    utterances are entirely disjoint. 
    :return: the waveform slices and mel spectrogram slices as lists of array slices. Index 
    respectively the waveform and the mel spectrogram with these slices to obtain the partial 
    utterances.
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


# def embed_frames_batch(frames_batch):
#     """
#     Computes embeddings for a batch of mel spectrogram.
    
#     :param frames_batch: a batch mel of spectrogram as a numpy array of float32 of shape 
#     (batch_size, n_frames, n_channels)
#     :return: the embeddings as a numpy array of float32 of shape (batch_size, model_embedding_size)
#     """
#     if _model is None:
#         raise Exception("Model was not loaded. Call load_model() before inference.")
    
#     frames = torch.from_numpy(frames_batch).to(_device)
#     embed = _model.forward(frames).detach().cpu().numpy()
#     return embed



```