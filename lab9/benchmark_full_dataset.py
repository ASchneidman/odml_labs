import sys, os
sys.path.append(os.path.join('..', 'lab7'))
sys.path.append(os.path.join('..', 'lab4'))

import matplotlib.pyplot as plt
import argparse
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

from timeit import default_timer as timer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    method_table = {
        'binary_lsh': {'enroll': enroll_lsh, 'query': query_lsh},
        'binary_shuffling': {'enroll': enroll_binary_shuffling, 'query': query_binary_shuffling},
        'random_projections': {'enroll': enroll_random_proj, 'query': query_random_proj},
        'binary_nn': {'enroll': enroll_bnn, 'query': query_bnn}
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling_rate", required=True, type=float)
    parser.add_argument("--secure_method", required=False, default="nonsecure", type=str)
    parser.add_argument("--relative_accuracy", required=True, action="store_true")
    args = parser.parse_args()

    sample_rate = args.sampling_rate

    audio_path= "../voxconverse"
    files = os.listdir(audio_path+"/audio")

    encoder = VoiceEncoder("cpu")


    average_time = 0.0
    average_rel_acc = 0.0
    num_files = 0

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

        # cut out the template segments
        speaker_wavs = [wav[int(s[0] * sample_rate):int(s[1] * sample_rate)] for s in templates.values()]

        # embed each segment
        speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]


        # start timer
        start_time = timer()

        # enroll each speaker
        if args.secure_method != 'nonsecure':
            # Enroll users
            speaker_passwords = list(templates.keys())
            print(f"Passwords: {speaker_passwords}")

            enrollments = []
            for i in range(len(speaker_passwords)):
                name, password, feature = list(templates.keys())[i], speaker_passwords[i], speaker_embeds[i]
                method_table[args.secure_method]['enroll'](name, password, feature, enrollments)

        # embed the entire wav
        _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True)

        # identify the speaker in each segment
        identified_speaker = []
        if args.secure_method == 'nonsecure':
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
                lowest_e = method_table[args.secure_method]['query'](embedding, enrollments)
                identified_speaker.append(lowest_e.name)

        end_time = timer()
        num_files += 1

        average_time += (end_time - start_time)

        if args.relative_accuracy:
            identified_speaker_nonsecure = []
            similarity_dict = {name: (cont_embeds @ speaker_embed).reshape(-1, 1) for name, speaker_embed in 
                            zip(list(templates.keys()), speaker_embeds)}
            name_indices = list(similarity_dict.keys())
            similarities = np.concatenate(list(similarity_dict.values()), axis=1)
            for i in range(len(wav_splits)):
                identified_speaker_nonsecure.append(name_indices[np.argmax(similarities[i])])

            accuracy = 0.0
            for speaker, speaker_nonsecure in zip(identified_speaker, identified_speaker_nonsecure):
                accuracy += float(speaker == speaker_nonsecure)

            accuracy /= len(identified_speaker)
            average_rel_acc += accuracy
            print(f"Relative Accuracy: {accuracy}")

    average_time /= num_files
    average_rel_acc /= num_files
    print(f"Average time: {num_files}, Average Rel. Accuracy {average_rel_acc}")

