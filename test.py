from autovc.utils.audio import get_mel_frames, audio_to_melspectrogram
from autovc.utils.model_loader import load_model
import torch
import numpy as np
import matplotlib.pyplot as plt
AE = load_model('auto_encoder', 'models/AutoVC/AutoVC_SMK.pt')
SE = load_model('speaker_encoder', 'models/SpeakerEncoder/SpeakerEncoder.pt')
N = 160
frames = get_mel_frames('data/conversions/hej.wav', audio_to_melspectrogram, sr = 22050, mel_window_step = 12.5, order = 'MF', partial_utterance_n_frames = N)
T = torch.stack(frames)
emb = torch.randn((T.size(0), 256))

_, out, _ = AE(T, emb, emb)

window = np.hanning(N)