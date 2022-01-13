from autovc.utils.dataloader import SpeakerEncoderDataLoader
import torch
from autovc.speaker_encoder.model import SpeakerEncoder
from autovc.utils.model_loader import load_model
import wandb
dataset = TrainDataLoader(speaker_encoder = self.SE, chop = False, data_path = 'data/test_data')

datadir = {'hilde': ['data/hilde_7sek'], 'hague': ['data/HaegueYang_10sek', 'data/hyang_smk']}
Data = SpeakerEncoderDataLoader(datadir)


SE = load_model('speaker_encoder', 'models/SpeakerEncoder/SpeakerEncoder.pt')
run = wandb.init(project = 'SpeakerEncoder',  entity = "deep_voice_inc", reinit = True)


dataloader = Data.get_dataloader(batch_size = 5)

SE.learn(dataloader, n_epochs = 100, wandb_run = run, )