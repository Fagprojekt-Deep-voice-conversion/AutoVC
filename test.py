from autovc.utils.audio import get_mel_frames, audio_to_melspectrogram
from autovc.utils.model_loader import load_model
import torch
import numpy as np
import matplotlib.pyplot as plt
# AE = load_model('auto_encoder', 'models/AutoVC/AutoVC_SMK.pt')


SE = load_model('speaker_encoder', 'models/SpeakerEncoder/SpeakerEncoder.pt')

SE.learn_speaker('hilde', 'data/samples')
SE.speakers
print(SE.speakers)
