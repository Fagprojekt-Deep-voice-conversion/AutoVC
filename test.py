# from autovc.speaker_encoder.model import SpeakerEncoder
# from autovc.speaker_encoder.utils import wav_to_mel_spectrogram, preprocess_wav
# import torch
# speaker_encoder = SpeakerEncoder()
# path = "data/samples/"
# data = [path + "chooped7" + ".wav", path + "conversion2" + ".wav", path + "mette_183" + ".wav" , path + "conversion1" + ".wav"  ]
# speaker_encoder.load_model('Models/SpeakerEncoder/SpeakerEncoder.pt')
# mels = [speaker_encoder(torch.from_numpy(wav_to_mel_spectrogram(preprocess_wav(wav))).unsqueeze(0)) for wav in data]
# l  = speaker_encoder.loss(torch.stack(mels))
# print(l)

import os
from autovc.speaker_encoder.utils import *
import soundfile as sf
from autovc.utils.hparams import SpeakerEncoderParams as hparams
from autovc.utils.preprocess_wav import audio_to_melspectrogram
# data_dir_path = 'data\\samples'

# [os.path.join(dirpath, filename) for dirpath , _, directory in os.walk(data_dir_path) for filename in directory]

# walk = [w for w in os.walk("data/yang_test")][0]
# root, dirs, files = [w for w in walk]

# for file in files:
#     waveform = preprocess_wav(os.path.join(root, file))
#     sf.write(f"test_yang_silence/{file}", np.asarray(waveform), samplerate = 16000)

walk = [w for w in os.walk("test_yang_silence")][0]
root, dirs, files = [w for w in walk]

for file in files:
    waveform = preprocess_wav(os.path.join(root, file))
    # sf.write(f"test_yang_silence/{file}", np.asarray(waveform), samplerate = 16000)