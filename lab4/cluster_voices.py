from scipy.spatial import distance
import sklearn
import s3prl.hub as hub
import torch
import torchaudio
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import argparse
import numpy as np
from torchaudio.transforms import Resample

from sklearn.cluster import KMeans

def modified_load(old_load):
    def new_load(*args, **kwargs):
        kwargs['map_location'] = torch.device('cpu')
        return old_load(*args, **kwargs)

    torch.load = new_load

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # name of files 
    parser.add_argument("input_wav", type=str)
    parser.add_argument("--model_name", type=str)
    #parser.add_argument("--downstream_weights", type=str)
    args = parser.parse_args()

    old_load = torch.load
    modified_load(old_load)
    model = getattr(hub, args.model_name)()
    model.eval()

    # load wav
    wav, freq = torchaudio.load(args.input_wav)
    print(wav.shape)

    # downsample
    resampler = Resample(freq, 16000)
    wav = resampler(wav)

    print(wav.shape)

    if wav.shape[0] == 2:
        wav = wav[0]

    with torch.no_grad():
        embeddings = model([wav])['hidden_states']


    embeddings = torch.cat([e[0] for e in embeddings])
    print(embeddings.shape)
    embeddings = embeddings.numpy()

    clusters = KMeans(10).fit(embeddings)

    plt.scatter(np.arange(embeddings.shape[0]), clusters.labels_)
    plt.show()

    #distances = cdist(embeddings, embeddings)
    #plt.imshow(distances)
    #plt.show()

