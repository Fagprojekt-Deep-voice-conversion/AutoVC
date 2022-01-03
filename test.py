from autovc.conversion import *
from autovc.zero_shot import *
from autovc.preprocessing import *
from autovc.preprocessing.preprocess_wav import *
from autovc.speaker_encoder.audio import preprocess_wav
import torch
from autovc.utils.dataloader2 import TrainDataLoader
from autovc.speaker_encoder.model import *
from autovc.auto_encoder.model_vc import Generator
if __name__ == "__main__":
    device = 'cpu'
    model = Generator(32, 256, 512, 32).to(device)
    pretrained_model_path = 'Models/AutoVC/autovc_origin.ckpt'
    g_checkpoint = torch.load(pretrained_model_path, map_location=torch.device(device))
    model.load_state_dict(g_checkpoint['model'])
    S = SpeakerEncoder(device = device, mel_n_channels = 80)
    
    datareader = TrainDataLoader('data')
    dataloader = datareader.data_loader(batch_size = 2)


    for mel, emb in dataloader:
        model(mel, emb, emb)
        print(mel.shape)

    # model, voc_model = Instantiate_Models(model_path = 'Models/AutoVC/AutoVC_SMK.pt')
    # batch = ["data/samples/mette_183.wav","data/samples/chooped7.wav"]
    # mels = compute_batch(batch)
    # m = WaveRNN_Mel("data/samples/mette_183.wav")
    # m = torch.Tensor(m).unsqueeze(0)
    # waveform = voc_model.generate(m, batched = True, target = 16_000, overlap = 550, mu_law= False)
    # sf.write(fpath + '.wav', np.asarray(waveform), samplerate = hp.sample_rate)
    # zero_shot("data/samples/mette_183.wav","data/samples/chooped7.wav", model, voc_model, ".")