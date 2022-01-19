from autovc.utils.model_loader import load_model
from autovc.utils.audio import audio_to_melspectrogram, get_mel_frames, remove_noise, normalize_volume
import torch

import soundfile as sf
import numpy as np

vocoder = load_model('vocoder', 'models/WaveRNN/WaveRNN_Pretrained.pyt')
AE = load_model('auto_encoder', 'models/AutoVC/AutoVC_SMK.pt')
SE = load_model('speaker_encoder', 'models/SpeakerEncoder/SpeakerEncoder.pt')
# SE.learn_speaker('HaegueYang', 'data/test_data/HaegueYang_10sek')

# SE = load_model('speaker_encoder', 'SpeakerEncoder3.pt')

wav = 'HYLydspor/HY1.wav'


mel_frames = get_mel_frames(wav,
                            audio_to_melspectrogram,
                            min_pad_coverage=0.75,
                            order = 'MF',
                            sr = 22050, 
                            mel_window_step = 12.5, 
                            partial_utterance_n_frames = 400,
                            overlap = 0.5)
                                    
batch = torch.stack(mel_frames)
print(batch.shape)
c_scr = SE.embed_utterance('HYLydspor/HY1.wav')
c_trg = SE.embed_utterance('data/test_data/HaegueYang_10sek/HaegueYang_0.wav')
# c_trg = SE.speakers['HaegueYang']
X = AE.batch_forward(batch[0].unsqueeze(0), c_scr, c_trg, overlap = 0.5 )


waveform = vocoder.generate(X.unsqueeze(0))

waveform = np.asarray(waveform)
waveform = remove_noise(waveform, 22050)
waveform = normalize_volume(waveform, -10)
sf.write('HowLong.wav', waveform, samplerate = 22050)
