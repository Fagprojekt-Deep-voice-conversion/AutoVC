from conversion import *
from Preprocessing_WAV import WaveRNN_Mel, AutoVC_Mel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# source = "data/samples/HaegueYang_5.wav" # file to run through vocoder
# source = "data/samples/aaa_z0031_052.wav"
source = "data/SMK_speakers/hyang_smk/aaa_z0030_011.wav"

source = WaveRNN_Mel(source)
source = torch.from_numpy(source.T).unsqueeze(0).to(device)

model, voc_model = Instantiate_Models(model_path = 'Models/AutoVC/AutoVC_SMK_20211104_original_step42.05k.pt')

Generate(source, "vocoder_out.wav", voc_model)