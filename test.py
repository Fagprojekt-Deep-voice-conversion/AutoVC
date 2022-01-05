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
data_dir_path = 'data\\samples'

[os.path.join(dirpath, filename) for dirpath , _, directory in os.walk(data_dir_path) for filename in directory]
