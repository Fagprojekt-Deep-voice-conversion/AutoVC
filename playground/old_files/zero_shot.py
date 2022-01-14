from conversion import *
import torch
from autovc.utils.audio import audio_to_melspectrogram
import matplotlib.pyplot as plt
import numpy as np
from autovc.wavernn.synthesizer import synthesize

def zero_shot(source, target, model, voc_model, save_path, name_path = None, only_conversion = True):
    """
    params:
    source: filepath to source file
    target: filepath to target file
    model: AutoVC model (use Instantiate_Models)
    voc_model: Vocder model (use Instantiate_Models)
    save_path: path to directory to store output
    only_conversion: only outputs converted file. If false source and target are outputted as well
    """

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    s = audio_to_melspectrogram(source)
    t = audio_to_melspectrogram(target)
   
    S, T = torch.from_numpy(s.T).unsqueeze(0).to(device), torch.from_numpy(t.T).unsqueeze(0).to(device)
    
    S_emb, T_emb = embed(source).to(device), embed(target).to(device)
    
    conversions = {"source": (S, S_emb, S_emb), "Converted": (S, S_emb, T_emb), "target": (T, T_emb, T_emb)}
    

    for key, (X, c_org, c_trg) in conversions.items():
        if key == "Converted":
            _, Out, _ = model(X, c_org, c_trg)
            name = f"{save_path}/{key}"
            print(f"\n Generating {key} sound")
            if name_path is not None:
                synthesize(Out, f"{save_path}/{name_path}", voc_model)

            else:
                synthesize(Out, name, voc_model)
        else:
            Out = X.unsqueeze(0)
            if not only_conversion:
                name = f"{save_path}/{key}"
                print(f"\n Generating {key} sound")
                synthesize(Out, name, voc_model)

# model, voc_model = Instantiate_Models(model_path = 'Models/AutoVC/autoVC30min_step72.pt')
if __name__ == "__main__":
    model, voc_model = Instantiate_Models(model_path = '../models/AutoVC/AutoVC_SMK.pt')
    
    zero_shot("../data/samples/mette_183.wav","../data/samples/chooped7.wav", model, voc_model, ".")


    
    