from scipy.spatial import distance
import sklearn
import s3prl.hub as hub
import torch
import torchaudio
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import argparse
import numpy as np
from torchaudio.transforms import Resample
from timeit import default_timer as timer


from sklearn.cluster import KMeans
import simpleder
from  utils_labels import *
import csv
import torch
def modified_load(old_load):
    def new_load(*args, **kwargs):
        kwargs['map_location'] = torch.device('cpu')
        return old_load(*args, **kwargs)

    torch.load = new_load


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # name of files 
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()

    models = [
        'wav2vec', 
        'wav2vec2_xlsr', 
        'vq_wav2vec', 
        'modified_cpc', 
        'wav2vec2', 
        'hubert'
    ]

    #device = 'cuda' # or cpu
    device = 'cpu'

    if args.model_name in models:
        old_load = torch.load
        modified_load(old_load)
        model = getattr(hub, args.model_name)().to(device)
        model.eval()
    
    # Load the resemblyzer model
    # TODO

    print('Model Loaded')

    # load wav
    #wav, freq = torchaudio.load("bdb001interaction_qAd8wbQp.wav")

    audio_path= "../../vox_converse_data"
    files = os.listdir(audio_path+"/audio")
    der_dict={}
    file_count=0

    #iterate through each audio file and it's annotation file in the dataset
    for file in files:
        basefile_name = file[0:-4]
        annotations_filepath= audio_path+"/annotations/"+basefile_name+".rttm"
        gold_labels = get_goldlabels(annotations_filepath)
        #print(gold_labels)
        wav_fpath = audio_path+"/audio/"+basefile_name+".wav"
        wav, freq = torchaudio.load(wav_fpath)
        print("freq:",freq)
        #wav, freq = torchaudio.load("quickbrownfox.wav")

        # Downsample the audio
        print(wav.shape)
        resampler = Resample(freq, 16000)
        wav = resampler(wav)
        print(wav.shape)

        if wav.shape[0] <= 2:
            wav = wav[0]

        start = timer()
        # Creates the embeddings
        with torch.no_grad():
            print('Creating embedding')
            print(wav.shape)
            if args.model_name in models:
                # Create embedding with one of the s3prl models
                embeddings = model([wav])['hidden_states']
            else:
                # Create embedding with Resemblyzer
                embeddings = model([wav])['hidden_states']
        embed_end = timer()

        embeddings = torch.cat([e[0] for e in embeddings])
        embeddings = embeddings.numpy()
        print("Shape of embeddings is", embeddings.shape)

        clusters = KMeans(4).fit(embeddings)
        cluster_end = timer()


        print('Running clustering')
        labels_preds = clusters.labels_
        labels_preds= postprocess_pred_labels(labels_preds)
        try:
            error = simpleder.DER(gold_labels, labels_preds)
        except ValueError:
            # skip this one
            continue
        print("file : {} DER={:.3f}".format(basefile_name,error))
        #plt.scatter(np.arange(embeddings.shape[0]), clusters.labels_)
        #plt.show()

    # calculate avg der of all the audio files
    sum =0

    for key,val in der_dict.items():

        print("DER for file {} = {:.3f}".format(key,val))
        sum += val
    avg_der= sum/len(der_dict)
    der_dict["total_avg_der"]= avg_der
    der_dict["inference_time"] = embed_end - start
    der_dict["diarization_time"] = cluster_end - start

    print('Writing results to file')
    f = open("results/"+args.model_name+"_der.csv","w")
    w = csv.writer(f)
    w.writerows(der_dict.items())
    w.close()
    print("AVG DER={:.3f}, Embedding Time={:.3f}, Clustering Time={.3f}".format(avg_der, embed_end-start, cluster_end))

 
