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

    audio_path= "../voxconverse"
    files = os.listdir(audio_path+"/audio")

    encoder = VoiceEncoder("cpu")

    num_files = 0
    parameters = list(itertools.product(list(method_table.keys()), sample_rates))
    stats = {p: {} for p in parameters}

    #iterate through each audio file and it's annotation file in the dataset
    for file in files:
        if '.wav' not in file:
            continue
        basefile_name = file[0:-4]
        annotations_filepath= audio_path+"/dev/"+basefile_name+".rttm"
        gold_labels = get_goldlabels(annotations_filepath)

        speakers = set([l[0] for l in gold_labels])
        if len(speakers) < 2:
            print(f"Less than 2 speakers in file {file}. Skipping this one.")
            continue

        templates = {}
        # pick out segments to use as templates
        for (speaker, start_time, end_time) in gold_labels:
            if speaker not in templates:
                if end_time - start_time > 5.0:
                    templates[speaker] = [start_time, end_time]

        if len(templates) != len(speakers):
            print(f"Could not find templates for all speakers in file {file}. Skipping this one.")
            continue

        # preprocess the wav file
        wav_fpath = audio_path+"/audio/"+basefile_name+".wav"
        wav = preprocess_wav(wav_fpath)

        # embed the entire wav
        _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True)

        # Now evaluate each method on this file
        for method_name, sampling_rate in parameters:
            parameter_stats = {'accuracy': [], 'time': []}
            
            print(f"Sample Rate: {sampling_rate}, Method Name: {method_name}")

            for _ in range(NUM_ITERS):
                # start timer
                start_time = timer()

                # cut out the template segments
                speaker_wavs = [wav[int(s[0] * sampling_rate):int(s[1] * sampling_rate)] for s in templates.values()]

                # embed each segment
                speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]

                # enroll each speaker
                if method_name != 'nonsecure':
                    # Enroll users
                    speaker_passwords = list(templates.keys())

                    enrollments = []
                    for i in range(len(speaker_passwords)):
                        name, password, feature = list(templates.keys())[i], speaker_passwords[i], speaker_embeds[i]
                        method_table[method_name]['enroll'](name, password, feature, enrollments)


                # identify the speaker in each segment
                identified_speaker = []
                if method_name == 'nonsecure':
                    similarity_dict = {name: (cont_embeds @ speaker_embed).reshape(-1, 1) for name, speaker_embed in 
                                    zip(list(templates.keys()), speaker_embeds)}
                    name_indices = list(similarity_dict.keys())
                    similarities = np.concatenate(list(similarity_dict.values()), axis=1)
                    for i in range(len(wav_splits)):
                        identified_speaker.append(name_indices[np.argmax(similarities[i])])
                else:
                    # Query each feature vector securely
                    predicted_speakers_secure = []
                    for embedding, split in zip(cont_embeds, wav_splits):
                        lowest_e = method_table[method_name]['query'](embedding, enrollments)
                        identified_speaker.append(lowest_e.name)

                end_time = timer()
                time_taken = (end_time - start_time)

                # Compute nonsecure identified speakers to compare against to compute relative accuracy
                identified_speaker_nonsecure = []
                similarity_dict = {name: (cont_embeds @ speaker_embed).reshape(-1, 1) for name, speaker_embed in 
                                zip(list(templates.keys()), speaker_embeds)}
                name_indices = list(similarity_dict.keys())
                similarities = np.concatenate(list(similarity_dict.values()), axis=1)
                for i in range(len(wav_splits)):
                    identified_speaker_nonsecure.append(name_indices[np.argmax(similarities[i])])

                # Compute relative accuracy
                accuracy = 0.0
                for speaker, speaker_nonsecure in zip(identified_speaker, identified_speaker_nonsecure):
                    accuracy += float(speaker == speaker_nonsecure)

                accuracy /= len(identified_speaker)
                print(f"Relative Accuracy: {accuracy}")

                parameter_stats['accuracy'].append(accuracy)
                parameter_stats['time'].append(time_taken)

            stats[(method_name, sampling_rate)][file] = parameter_stats


    with open('stats.pkl', 'wb') as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)