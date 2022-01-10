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


def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    if isinstance(wav, str) or isinstance(wav, Path):
        wav, source_sr = librosa.load(wav, sr=None)

    frames = librosa.feature.melspectrogram(
        wav,
        hparams.sampling_rate,
        n_fft=int(hparams.sampling_rate * hparams.mel_window_length / 1000),
        hop_length=int(hparams.sampling_rate * hparams.mel_window_step / 1000),
        n_mels=hparams.mel_n_channels
    )

    return frames.astype(np.float32).T



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


