from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from pathlib import Path
import numpy as np
from random_projections import enroll, query
from timeit import default_timer as timer
import matplotlib.pyplot as plt

## Get reference audios
# Source for the interview: https://www.youtube.com/watch?v=X2zqiX6yL3I
wav_fpath = "uef39_3.mp4"
wav = preprocess_wav(wav_fpath)

secure_inf_time = []
nonsecure_inf_time = []
accuracies = []

# Compute times and accuracies when varying sampling rate
sampling_rates = np.arange(1000, 17000, 1000)
for i in sampling_rates:

    sample_rate = sampling_rates[i]

    # Cut some segments from single speakers as reference audio
    segments = [[0, 5.5], [6.5, 12], [17, 25]]
    speaker_names = ["Kyle Gass", "Sean Evans", "Jack Black"]
    speaker_wavs = [wav[int(s[0] * sample_rate):int(s[1] * sample_rate)] for s in segments]
      
    encoder = VoiceEncoder("cpu")
    speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]


    # Enroll users
    speaker_passwords = speaker_names
    print(f"Passwords: {speaker_passwords}")

    enrollments = []
    for i in range(len(speaker_passwords)):
        name, password, feature = speaker_names[i], speaker_passwords[i], speaker_embeds[i]
        enroll(name, password, feature, enrollments)


    print("Running the continuous embedding on cpu, this might take a while...")
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
    # Query feature vector non-securely
    non_secure_q_start = timer()
    predicted_speakers_not_secure = []
    for embedding, split in zip(cont_embeds, wav_splits):

        best_sim = float("-inf")
        best_name = None
        for name, speaker_embed in zip(speaker_names, speaker_embeds):
            sim = embedding @ speaker_embed
            
            if sim > best_sim:
                # Note, best similarity here is the LARGEST, not smallest
                best_sim = sim
                best_name = name

        predicted_speakers_not_secure.append(best_name)
    non_secure_q_end = timer()
    nonsecure_inf_time.append(non_secure_q_end - non_secure_q_start)


    # Query each feature vector securely
    secure_q_start = timer()
    predicted_speakers_secure = []
    for embedding, split in zip(cont_embeds, wav_splits):
        lowest_e = query(embedding, enrollments)
        predicted_speakers_secure.append(lowest_e.name)
    secure_q_end = timer()
    secure_inf_time.append(secure_q_end - secure_q_start)



    accuracy = 0.0
    # compute accuracy relative to non secure version
    for pred_secure, pred_not_secure in zip(predicted_speakers_secure, predicted_speakers_not_secure):
        accuracy += float(pred_secure == pred_not_secure)

    accuracy /= len(predicted_speakers_not_secure)

    print(f"Accuracy: {accuracy}")

    predicted_speakers_not_secure = [s + f", {split.start / sample_rate, split.stop / sample_rate}" + '\n' for s, split in zip(predicted_speakers_not_secure, wav_splits)]
    predicted_speakers_secure = [s + f", {split.start / sample_rate, split.stop / sample_rate}" + '\n' for s, split in zip(predicted_speakers_secure, wav_splits)]

    with open('secure_names_output.txt', 'w') as f:
        f.writelines(predicted_speakers_secure)
    with open('not_secure_names_output.txt', 'w') as f:
        f.writelines(predicted_speakers_not_secure)

    # Get the continuous similarity for every speaker. It amounts to a dot product between the 
    # embedding of the speaker and the continuous embedding of the interview
    similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in 
                       zip(speaker_names, speaker_embeds)}

np.savez('./metrics.npz', nonsecure_inf_time,
                          secure_inf_time,
                          accuracies)

plt.figure
X = sampling_rates
plt.plot(X, nonsecure_inf_time, label='Nonsecure Query')
plt.plot(X, secure_inf_time, label='Secure Query with Biometric Salting')
plt.title('Query Times vs. Sampling Rate for a Test Audio')
plt.ylabel('Query Times')
plt.xlabel('Audio Sampling Rate')
plt.grid()
plt.legend()
plt.savefig("inference_times.png")

plt.figure
plt.plot(X, accuracies, label='Diarization Accuracy')
plt.title('Inference Times vs. Input Size for 3 Audio Embedding Models')
plt.ylabel('Inference Times')
plt.xlabel('Input Size')
plt.grid()
plt.legend()

