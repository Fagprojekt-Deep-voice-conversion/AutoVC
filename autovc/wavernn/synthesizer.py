# import os,sys,inspect
# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir) 
import torch
# print(sys.path[0])
from autovc.utils.model_loader import load_vocoder


from autovc.utils.hparams import WaveRNNParams as hparams
import soundfile as sf
import numpy as np
from autovc.wavernn.model import WaveRNN
import os

# def load_model(model_path, **kwargs):
#     device = torch.device(kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
#     model = WaveRNN(**hparams.model.__dict__).to(device=device)
#     model.load(model_path)
#     return model

def synthesize(melspec, fpath = "results/synthesized.wav", model = None, **generate_args):
    """
    Synthesizes a melspectrogram

    Params 
    ------
    melspec: 
        a melspectrogram
    fpath: 
        file path to save the synthesized melspectrogram to
    model:
        a WaveRNN model 
    """
    # m = torch.tensor(melspec).squeeze(0)#.unsqueeze(0)
    m = melspec
    m = m.squeeze(0).squeeze(0).T.unsqueeze(0)
    
    device = generate_args.pop("device", "cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = load_vocoder('../models/WaveRNN/WaveRNN_Pretrained.pyt', device = device)
    elif isinstance(model, str):
        model = load_vocoder(model, device = device)
        
    waveform = model.generate(m, **generate_args)
    

    # save file
    # librosa.output.write_wav(fpath + '.wav', np.asarray(waveform), sr = hparams.sample_rate)
    folder, filename = os.path.split(fpath)
    os.makedirs(folder, exist_ok=True) # create folder
    fpath += '.wav' if os.path.splitext(fpath)[1] == '' else '' # add extension if missing
    sf.write(fpath, np.asarray(waveform), samplerate = hparams.sample_rate)

