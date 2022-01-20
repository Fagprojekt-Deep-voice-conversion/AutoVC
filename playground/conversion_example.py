
from playground.audio import audio_to_melspectrogram, remove_noise
# from autovc.speaker_encoder.model import SpeakerEncoder
# from autovc.auto_encoder.model_vc import Generator
# from autovc.wavernn.model import WaveRNN
import soundfile as sf
import torch
import numpy as np
from autovc.utils.model_loader import load_models

if __name__ == "__main__":
    # Models
    autovc_model = 'models/AutoVC/autovc_SMK.pt'
    speaker_encoder_model = "models/SpeakerEncoder/SpeakerEncoder.pt"
    vocoder_model = "models/WaveRNN/WaveRNN_Pretrained.pyt"
    device = 'cpu'

    # # AutoVC
    # model = Generator()

    # # SpeakerEncoder
    # S = SpeakerEncoder(device = device)

    # # vocoder
    # voc_model = WaveRNN().to(device)


    # # Load weights
    # voc_model.load(vocoder_model)
    # model.load_model(autovc_model, device)
    # S.load_model(speaker_encoder_model)

    model, S, voc_model = load_models(
        model_types= ["auto_encoder", "speaker_encoder", "vocoder"],
        model_paths= [autovc_model, speaker_encoder_model, vocoder_model],
        device = device
    )
    
    # Source and target
    target, source = "data/samples/chooped7.wav", "data/samples/mette_183.wav"

    # Generate speaker embeddings
    c_source, c_target = S.embed_utterance(source).unsqueeze(0), S.embed_utterance(target).unsqueeze(0)

    # Create mel spectrogram
    X = audio_to_melspectrogram(source)

    # Convert:
    #   out is the converted output
    #   post_out is the refined (by postnet) out put
    #   content_codes is the content vector - the content encoder output
    out, post_out, content_codes = model(torch.from_numpy(X).unsqueeze(0), c_source, c_target)
    
    # Use the Vocoder to generate waveform (use post_out as input)
    waveform = voc_model.generate(post_out)

    # reduce noise
    waveform = remove_noise(waveform, voc_model.params.sample_rate)

    # Generate .wav file frowm waveform
    sf.write('conversion1.wav', np.asarray(waveform), samplerate =voc_model.params.sample_rate)



    