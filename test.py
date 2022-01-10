from autovc.utils.model_loader import load_model
from autovc.utils.dataloader import SpeakerEncoderDataLoader
import torch
datadir = {'hilde': ['data/conversions'], 'hague': ['data/conversions2'], 'peter':['data/new']}
Data = SpeakerEncoderDataLoader(datadir)

dataloader = Data.get_dataloader()
SE = load_model('speaker_encoder', 'models/SpeakerEncoder/SpeakerEncoder.pt')


def batch_forward(batch):
    embeddings = []
    for b in batch:
        embed_speaker = torch.stack([SE.forward(torch.from_numpy(speaker).unsqueeze(0).to('cpu')) for speaker in b])
        embeddings.append(embed_speaker)
    return torch.cat(embeddings, dim = 1)


for i in range(5):
    for batch in dataloader:
        embeds = batch_forward(batch)
        print(embeds.shape)
        print(SE.similarity_matrix(embeds))