from scipy.ndimage.morphology import binary_dilation
from autovc.utils.hparams import SpeakerEncoderParams as hparams
from pathlib import Path
from typing import Optional, Union
import numpy as np
import webrtcvad
import librosa
import struct
import soundfile as sf
import os

int16_max = (2 ** 15) - 1
hparams = hparams()


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform's sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(fpath_or_wav, sr=None)
    else:
        wav = fpath_or_wav
    
    # Resample the wav if needed
    if source_sr is not None and source_sr != hparams.sampling_rate:
        wav = librosa.resample(wav, source_sr, hparams.sampling_rate)
        
    # Apply the preprocessing: normalize volume and shorten long silences 
    #wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    wav = trim_long_silences(wav)
    
    return wav


def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        wav,
        hparams.sampling_rate,
        n_fft=int(hparams.sampling_rate * hparams.mel_window_length / 1000),
        hop_length=int(hparams.sampling_rate * hparams.mel_window_step / 1000),
        n_mels=hparams.mel_n_channels
    )

    return frames.astype(np.float32).T


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


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))



def compute_partial_slices(n_samples, partial_utterance_n_frames=hparams.partials_n_frames,
                           min_pad_coverage=0.75, overlap=0.5):
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
    
    samples_per_frame = int((hparams.sampling_rate * hparams.mel_window_step / 1000))
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





def create_audio_mask(wav):
    """"
    Creates an audio mask, where False indicates silence.

    :param wav: a numpy array with the content of a wav file
    :return audio_mask: the calculated audio mask
    """
    # Compute the voice detection window size
    samples_per_window = (hparams.vad_window_length * hparams.sampling_rate) // 1000
    
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
                                         sample_rate=hparams.sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, hparams.vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(hparams.vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav, audio_mask

def split_audio(wav_path, save_folder = None, allowed_pause = .5):
    """
    Chops the content of the wav file into multiple files, based on when there is
    a longer period if silence. Files will be saved with a number indicating the order the content appeared in.

    :param wav: path to wav file
    :param save_folder: folder to save files in
    :allowed_pause: number seconds of silence to allow in a sound file
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
            if split[-1] - joined_splits[-1][-1] <= allowed_pause:
                joined_splits.append(np.concatenate([joined_splits.pop(), split]))
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




# split_audio("data/samples/mette_183.wav", "results")
