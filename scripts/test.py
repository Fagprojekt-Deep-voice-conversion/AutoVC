import gdown

waveRNN_folder = ("13_BgX2ZC-OOlGCJ5_-nIJUgQ-E-WtFsH", "WaveRNN")

# test = (1,2)

for url, output in list(zip(*zip(waveRNN_folder))):
    url = "https://drive.google.com/uc?id=" + url
    url = "https://drive.google.com/drive/u/0/folders/13_BgX2ZC-OOlGCJ5_-nIJUgQ-E-WtFsH"
    gdown.download_folder(url, output, quiet=False)