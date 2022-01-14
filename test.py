from autovc.utils.audio import get_mel_frames, audio_to_melspectrogram
from autovc.utils.model_loader import load_model
import torch


AE = load_model('auto_encoder', 'models/AutoVC/AutoVC_SMK.pt')

print(torch.stack(get_mel_frames('data/conversions/hej.wav', audio_to_melspectrogram, sr = 22000, order = 'MF')).shape)

