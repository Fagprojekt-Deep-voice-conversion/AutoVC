from conversion import *
import torch
from Preprocessing_WAV import WaveRNN_Mel, AutoVC_Mel
import matplotlib.pyplot as plt
import numpy as np

def Zero_shot(source, target, model, voc_model, save_path, name_path = None, only_conversion = True):
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

    s = WaveRNN_Mel(source)
    t = WaveRNN_Mel(target)
   
    S, T = torch.from_numpy(s.T).unsqueeze(0).to(device), torch.from_numpy(t.T).unsqueeze(0).to(device)
    
    S_emb, T_emb = embed(source).to(device), embed(target).to(device)
    
    conversions = {"source": (S, S_emb, S_emb), "Converted": (S, S_emb, T_emb), "target": (T, T_emb, T_emb)}
    

    for key, (X, c_org, c_trg) in conversions.items():
        if key == "Converted":
            _, Out, _ = model(X, c_org, c_trg)
            name = f"{save_path}/{key}"
            print(f"\n Generating {key} sound")
            if name_path is not None:
                Generate(Out, f"{save_path}/{name_path}", voc_model)

            else:
                Generate(Out, name, voc_model)
        else:
            Out = X.unsqueeze(0)
            if not only_conversion:
                name = f"{save_path}/{key}"
                print(f"\n Generating {key} sound")
                Generate(Out, name, voc_model)
            
        
        
        
        
        

# model, voc_model = Instantiate_Models(model_path = 'Models/AutoVC/autoVC30min_step72.pt')
if __name__ == "__main__":

    import shutil, os, re


    # model, voc_model = Instantiate_Models(model_path = 'Models/AutoVC/AutoVC_SMK.pt')
    # model, voc_model = Instantiate_Models(model_path = 'Models/AutoVC/AutoVC_SMK2_original_step20.85k.pt')

    model_paths = [
        'Models/AutoVC/AutoVC_SMK_20211104_original_step42.05k.pt',
        'Models/AutoVC/AutoVC_SMK_20211025_original_step29.275k.pt',
        'Models/AutoVC/AutoVC_SMK3_original_step26.825k.pt'
        ]
    sources = [
        "./data/samples/hilde_1.wav",
        "./data/SMK_speakers/hilde_7sek/hilde_18.wav",
        "./data/samples/hilde_301.wav",
        ]
    targets = [
        "./data/samples/HaegueYang_5.wav",
        "./data/SMK_speakers/HaegueYang_10sek/HaegueYang_8.wav",
        "./data/samples/HaegueSMK.wav"
        ]
    
    save_dir = "./data/conversion/"

    # model_path = model_paths[0]
    # source = sources[0]
    # target = targets[0]

    for model_path in model_paths:
        save_dir = "./data/conversion/" + re.findall(r"(.*)_original", model_path.split("/")[-1])[0] + "/"
        # for source, target in zip(sources, targets):
        for source in sources:
            for target in targets:
                model, voc_model = Instantiate_Models(model_path = model_path)
                Zero_shot(source,target, model, voc_model, ".")

                os.makedirs(save_dir, exist_ok=True)
                src_name = re.findall(r"(.*).wav", source.split("/")[-1])[0]
                trg_name = re.findall(r"(.*).wav", target.split("/")[-1])[0]
                shutil.copyfile(source, save_dir + src_name + "_source.wav")
                shutil.copyfile(target, save_dir + trg_name +  "_target.wav")
                shutil.copyfile("./conversion.wav", save_dir + f"./{src_name}_to_{trg_name}_conversion.wav")
    