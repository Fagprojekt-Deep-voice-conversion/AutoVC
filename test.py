from autovc.utils.model_loader import load_model
from autovc.utils.dataloader import SpeakerEncoderDataLoader
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
datadir = {'hilde': ['data/hyang_smk', 'data/hilde_20211020'], 'hague': ['data/HaegueYang_10sek', 'data/hyang_smk']}
Data = SpeakerEncoderDataLoader(datadir)

dataloader = Data.get_dataloader(batch_size=50)
SE = load_model('speaker_encoder', 'models/SpeakerEncoder/SpeakerEncoder.pt')


def batch_forward(batch):
    # embeddings = []
    # for b in batch:
    #     embed_speaker = torch.stack([SE.forward(torch.from_numpy(speaker).unsqueeze(0).to('cpu')) for speaker in b])
    #     embeddings.append(embed_speaker)
    # return torch.cat(embeddings, dim = 1)
    return torch.stack([SE(b) for b in batch])

for i in range(5):
    for batch in dataloader:
        embeds = batch_forward(batch)
        print(embeds.shape)
        X = TSNE(n_components=2 ).fit_transform(torch.flatten(embeds, start_dim  = 0, end_dim = 1).detach().numpy())
        plt.scatterplot(X[:, 0], X[:, 1])
        print(SE.loss(embeds))