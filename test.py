from autovc.conversion import *
from autovc.zero_shot import *
from autovc.preprocessing import *
from autovc.preprocessing.preprocess_wav import *
from autovc.speaker_encoder.audio import preprocess_wav
import torch
from autovc.utils.dataloader2 import TrainDataLoader
from autovc.Speaker_identity import *
if __name__ == "__main__":
    D = TrainDataLoader('data')
    D.data_loader(batch_size=2)
    [a for a in D.data_loader(batch_size = 2)]
    print(D.mel_spectograms)

    # model, voc_model = Instantiate_Models(model_path = 'Models/AutoVC/AutoVC_SMK.pt')
    # batch = ["data/samples/mette_183.wav","data/samples/chooped7.wav"]
    # mels = compute_batch(batch)
    # m = WaveRNN_Mel("data/samples/mette_183.wav")
    # m = torch.Tensor(m).unsqueeze(0)
    # waveform = voc_model.generate(m, batched = True, target = 16_000, overlap = 550, mu_law= False)
    # sf.write(fpath + '.wav', np.asarray(waveform), samplerate = hp.sample_rate)
    # zero_shot("data/samples/mette_183.wav","data/samples/chooped7.wav", model, voc_model, ".")