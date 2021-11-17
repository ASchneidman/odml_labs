from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from pathlib import Path
import numpy as np
from random_projections import enroll, query
from timeit import default_timer as timer



# DEMO 02: we'll show how this similarity measure can be used to perform speaker diarization
# (telling who is speaking when in a recording).


## Get reference audios
# Load the interview audio from disk
# Source for the interview: https://www.youtube.com/watch?v=X2zqiX6yL3I
wav_fpath = "uef39_3.mp4"
wav = preprocess_wav(wav_fpath)

# Cut some segments from single speakers as reference audio
segments = [[0, 5.5], [6.5, 12], [17, 25]]
speaker_names = ["Kyle Gass", "Sean Evans", "Jack Black"]
speaker_wavs = [wav[int(s[0] * sampling_rate):int(s[1] * sampling_rate)] for s in segments]
  
encoder = VoiceEncoder("cpu")
speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]


# Enroll users
speaker_passwords = speaker_names
print(f"Passwords: {speaker_passwords}")

enrollments = []
for i in range(len(speaker_passwords)):
    name, password, feature = speaker_names[i], speaker_passwords[i], speaker_embeds[i]

    enroll(name, password, feature, enrollments)


# Test that the templates match in the secure space
print("Test that the templates match in the secure space")
for name, speaker_embed in zip(speaker_names, speaker_embeds):
    print(f"Actual: {name}, Pred: {query(speaker_embed, enrollments).name}")

    
## Compare speaker embeds to the continuous embedding of the interview
# Derive a continuous embedding of the interview. We put a rate of 16, meaning that an 
# embedding is generated every 0.0625 seconds. It is good to have a higher rate for speaker 
# diarization, but it is not so useful for when you only need a summary embedding of the 
# entire utterance. A rate of 2 would have been enough, but 16 is nice for the sake of the 
# demonstration. 
# We'll exceptionally force to run this on CPU, because it uses a lot of RAM and most GPUs 
# won't have enough. There's a speed drawback, but it remains reasonable.

print("Running the continuous embedding on cpu, this might take a while...")

_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
# Query feature vector non-securely
non_secure_q_start = timer()
predicted_speakers_not_secure = []
for embedding, split in zip(cont_embeds, wav_splits):
    #print(f"Audio Segment: {split.start / sampling_rate, split.stop / sampling_rate}")

    best_sim = float("-inf")
    best_name = None
    for name, speaker_embed in zip(speaker_names, speaker_embeds):
        sim = embedding @ speaker_embed
        
        #print(f"Non-Secure: Name {name}, Dist: {sim}")
        if sim > best_sim:
            # Note, best similarity here is the LARGEST, not smallest
            best_sim = sim
            best_name = name

    predicted_speakers_not_secure.append(best_name)
non_secure_q_end = timer()


# Query each feature vector securely
secure_q_start = timer()
predicted_speakers_secure = []
for embedding, split in zip(cont_embeds, wav_splits):
    #print(f"Audio Segment: {split.start / sampling_rate, split.stop / sampling_rate}")
    lowest_e = query(embedding, enrollments)
    predicted_speakers_secure.append(lowest_e.name)
secure_q_end = timer()




accuracy = 0.0
# compute accuracy relative to non secure version
for pred_secure, pred_not_secure in zip(predicted_speakers_secure, predicted_speakers_not_secure):
    accuracy += float(pred_secure == pred_not_secure)

accuracy /= len(predicted_speakers_not_secure)

print(f"Accuracy: {accuracy}")

predicted_speakers_not_secure = [s + f", {split.start / sampling_rate, split.stop / sampling_rate}" + '\n' for s, split in zip(predicted_speakers_not_secure, wav_splits)]
predicted_speakers_secure = [s + f", {split.start / sampling_rate, split.stop / sampling_rate}" + '\n' for s, split in zip(predicted_speakers_secure, wav_splits)]

with open('secure_names_output.txt', 'w') as f:
    f.writelines(predicted_speakers_secure)
with open('not_secure_names_output.txt', 'w') as f:
    f.writelines(predicted_speakers_not_secure)

# Get the continuous similarity for every speaker. It amounts to a dot product between the 
# embedding of the speaker and the continuous embedding of the interview
similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in 
                   zip(speaker_names, speaker_embeds)}



## Run the interactive demo
interactive_diarization(similarity_dict, wav_fpath, wav_splits, show_time=True)


