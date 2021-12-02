import sys, os
sys.path.append(os.path.join('..', 'lab7'))

from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
import numpy as np

from random_projections import enroll as enroll_random_proj
from random_projections import query as query_random_proj

from binary_shuffling import enroll as enroll_binary_shuffling
from binary_shuffling import query as query_binary_shuffling

from timeit import default_timer as timer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sampling_rate", required=True, type=float)
parser.add_argument("--secure_method", required=False, default="nonsecure", type=str)
args = parser.parse_args()

## Get reference audios
# Source for the interview: https://www.youtube.com/watch?v=X2zqiX6yL3I
wav_fpath = os.path.join('..', os.path.join('lab7', "uef39_3.mp4"))
wav = preprocess_wav(wav_fpath)

secure_inf_time = []
nonsecure_inf_time = []
accuracies = []

sample_rate = args.sampling_rate

# Cut some segments from single speakers as reference audio
segments = [[0, 5.5], [6.5, 12], [17, 25]]
speaker_names = ["Kyle Gass", "Sean Evans", "Jack Black"]
speaker_wavs = [wav[int(s[0] * sample_rate):int(s[1] * sample_rate)] for s in segments]
    
encoder = VoiceEncoder("cpu")
speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]


if args.secure_method != 'nonsecure':
    # Enroll users
    speaker_passwords = speaker_names
    print(f"Passwords: {speaker_passwords}")

    enrollments = []
    for i in range(len(speaker_passwords)):
        name, password, feature = speaker_names[i], speaker_passwords[i], speaker_embeds[i]
        if args.secure_method == 'binary_shuffling':
            enroll_binary_shuffling(name, password, feature, enrollments)
        else:
            enroll_random_proj(name, password, feature, enrollments)


_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True)

timer_start = timer()

if args.secure_method == 'nonsecure':
    for embedding, split in zip(cont_embeds, wav_splits):
        best_sim = float("-inf")
        best_name = None
        for name, speaker_embed in zip(speaker_names, speaker_embeds):
            sim = embedding @ speaker_embed
            
            if sim > best_sim:
                # Note, best similarity here is the LARGEST, not smallest
                best_sim = sim
                best_name = name
else:
    # Query each feature vector securely
    predicted_speakers_secure = []
    for embedding, split in zip(cont_embeds, wav_splits):
        if args.secure_method == 'binary_shuffling':
            lowest_e = query_binary_shuffling(embedding, enrollments)
        else:
            lowest_e = query_random_proj(embedding, enrollments)

timer_end = timer()

print(f"Time taken: {timer_end - timer_start}")
