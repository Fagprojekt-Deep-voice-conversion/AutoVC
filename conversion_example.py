
from autovc.preprocessing.preprocess_wav import WaveRNN_Mel
from autovc.speaker_encoder.model import SpeakerEncoder
from autovc.auto_encoder.model_vc import Generator
from autovc.vocoder.WaveRNN_model import WaveRNN
from autovc.utils.hparams import hparams_waveRNN as hp
import soundfile as sf
import torch
import numpy as np

if __name__ == "__main__":
    # Models
    autovc_model = 'Models/AutoVC/autovc_SMK.pt'
    speaker_encoder_model = "Models/SpeakerEncoder/SpeakerEncoder.pt"
    vocoder_model = "Models/WaveRNN/WaveRNN_Pretrained.pyt"
    device = 'cpu'

    # AutoVC
    model = Generator()

    # SpeakerEncoder
    S = SpeakerEncoder(device = device)

    # Vocode - WaveRNN
    voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                            fc_dims=hp.voc_fc_dims,
                            bits=hp.bits,
                            pad=hp.voc_pad,
                            upsample_factors=hp.voc_upsample_factors,
                            feat_dims=hp.num_mels,
                            compute_dims=hp.voc_compute_dims,
                            res_out_dims=hp.voc_res_out_dims,
                            res_blocks=hp.voc_res_blocks,
                            hop_length=hp.hop_length,
                            sample_rate=hp.sample_rate,
                            mode='MOL').to(device)

    # Load weights
    voc_model.load(vocoder_model)
    model.load_model(autovc_model, device)
    S.load_model(speaker_encoder_model)
    
    # Source and target
    source, target = "data/samples/chooped7.wav", "data/samples/mette_183.wav"

    # Generate speaker embeddings
    c_source, c_target = S.embed_utterance(source).unsqueeze(0), S.embed_utterance(target).unsqueeze(0)

    # Create mel spectrogram
    X = WaveRNN_Mel(target)

    # Convert:
    #   out is the converted output
    #   post_out is the refined (by postnet) out put
    #   content_codes is the content vector - the content encoder output
    out, post_out, content_codes = model(torch.from_numpy(X).unsqueeze(0), c_source, c_target)
    
    # Use the Vocoder to generate waveform (use post_out as input)
    waveform = voc_model.generate(post_out)

    # Generate .wav file frowm waveform
    sf.write('conversion11.wav', np.asarray(waveform), samplerate = hp.sample_rate)



    