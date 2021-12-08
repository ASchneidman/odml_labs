import sys, os
sys.path.append(os.path.join('..', 'lab7'))
sys.path.append(os.path.join('..', 'lab4'))

import itertools
import numpy as np
from timeit import default_timer as timer

from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *

from random_projections import enroll as enroll_random_proj
from random_projections import query as query_random_proj

from binary_shuffling import enroll as enroll_binary_shuffling
from binary_shuffling import query as query_binary_shuffling

from binary_lsh import enroll as enroll_lsh
from binary_lsh import query as query_lsh

from binary_nearest_neighbors import enroll as enroll_bnn
from binary_nearest_neighbors import query as query_bnn

from  utils_labels import *
import csv
import pickle

from timeit import default_timer as timer

NUM_ITERS = 5

if __name__ == "__main__":
    method_table = {
        'binary_lsh': {'enroll': enroll_lsh, 'query': query_lsh},
        'binary_shuffling': {'enroll': enroll_binary_shuffling, 'query': query_binary_shuffling},
        'random_projections': {'enroll': enroll_random_proj, 'query': query_random_proj},
        'binary_nn': {'enroll': enroll_bnn, 'query': query_bnn},
        'nonsecure': None
    }

    sample_rates = np.arange(2000, 18000, 2000)

    start_time = timer()

    encoder = VoiceEncoder("cpu")

    num_files = 0
    parameters = list(itertools.product(list(method_table.keys()), sample_rates))
    stats = {p: {} for p in parameters}

    # preprocess the wav file
    wav_fpath = os.path.join('..', os.path.join('lab7', "uef39_3.mp4"))
    wav = preprocess_wav(wav_fpath)

    # Cut some segments from single speakers as reference audio
    segments = [[0, 5.5], [6.5, 12], [17, 25]]
    speaker_names = ["Kyle Gass", "Sean Evans", "Jack Black"]


    # embed the entire wav
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True)

    end_time = timer()

    print(f"Time to preprocess: {end_time - start_time}")