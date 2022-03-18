# from autovc.models import load_model
# # from playground.audio import audio_to_melspectrogram, get_mel_frames, remove_noise, normalize_volume
# import torch
# from autovc import audio
# import soundfile as sf
# import numpy as np

# vocoder = load_model('vocoder', 'WaveRNN_Pretrained.pyt')
# AE = load_model('auto_encoder', 'SMK2.pt')
# SE = load_model('speaker_encoder', 'SpeakerEncoder_SMK2.pt')


# wav = 'data/HYLydspor/HY2.wav'

# data = audio.Audio(wav, sr = 22050)
# data.preprocess("normalize_volume", target_dBFS = -20)
# mel_frames = audio.spectrogram.mel_spec_auto_encoder(data.wav, 
#                                                     sr = data.sr,
#                                                     cut = True,
#                                                     min_pad_coverage=0.75,
#                                                     # order = 'MF', 
#                                                     mel_window_step = 12.5, 
#                                                     partial_utterance_n_frames = 200,
#                                                     overlap = 0.5)


                                    
# batch = torch.stack(mel_frames)
# c_scr = SE.speakers['louise']
# c_trg = SE.speakers['yang']

# X = AE.batch_forward(batch[0].unsqueeze(0), c_scr, c_trg, overlap = 0.5 )


# waveform = vocoder.generate(X.unsqueeze(0))

# waveform = np.asarray(waveform)
# waveform = audio.tools.remove_noise(waveform, 22050)
# waveform = audio.tools.normalize_volume(waveform, -20)
# sf.write('Conversion.wav', waveform, samplerate = 22050)
from autovc.voice_converter import VoiceConverter

VC = VoiceConverter(auto_encoder='SMK_train2.pt', speaker_encoder='SpeakerEncoder_SMK2.pt')

wavs = ["HY1", "HY2", "HY3", "HY4", "HY4a", "HY4b", "HY4c", "HY5", "HY6", "HY7", "HY8","HY9" ]



for wav in wavs:
    VC.convert(source = 'data/HYLydspor/' + wav + '.wav', target = 'yang', save_name = wav + '_converted.wav',
            preprocess = ["normalize_volume"],
            preprocess_args = {"target_dBFS" : -20},
            outprocess = ["normalize_volume", "remove_noise"],
            outprocess_args = {"target_dBFS" : -20}, 
            device = 'cuda',
            cut = True,
            min_pad_coverage=0.75,
            sr = 22050, 
            mel_window_step = 12.5, 
            partial_utterance_n_frames = 150,
            overlap = 0.5)