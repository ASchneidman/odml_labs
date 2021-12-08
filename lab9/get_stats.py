import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import librosa

POWER_FACTOR = 7.2
BUDGET = 35693.2

#audio_path = os.path.join('..', os.path.join('voxconverse', 'audio'))
audio_path = '../lab7'

with open('interview_stats.pkl', 'rb') as f:
    stats = pickle.load(f)

sample_rates = sorted(set([s[1] for s in stats.keys()]))
method_names = set([s[0] for s in stats])

sample_rate = 16000

plt.figure()

ys = []
err = []
max_seconds_audio = []
relative_accuracy = []

for name in method_names:
    normalized_inference_times = []
    for file_name in stats[(name, sample_rate)]:
        duration = librosa.get_duration(filename=os.path.join(audio_path, f"{file_name}"))
        for t in stats[(name, sample_rate)][file_name]['time']:
            normalized_inference_times.append(t / duration)

    normalized_inference_times = np.array(normalized_inference_times) * POWER_FACTOR
    ys.append(np.array(normalized_inference_times).mean())
    err.append(np.array(normalized_inference_times).std())

    acc = np.array([d['accuracy'] for d in stats[(name, sample_rate)].values()]).flatten()

    relative_accuracy.append(acc.mean())
    max_seconds_audio.append(BUDGET / ys[-1])

x_pos = [i for i, _ in enumerate(method_names)]

plt.barh(x_pos, ys, color='green', xerr=err)
plt.ylabel("Method")
plt.xlabel("Watt Seconds Per Second of Audio")
plt.title("Watt Seconds Per Second of Audio by Secure Method")

plt.yticks(x_pos, method_names)

plt.show()


fig, ax = plt.subplots()
max_hours_audio = np.array(max_seconds_audio) / 36000.0
ax.scatter(max_hours_audio, relative_accuracy)

for i, txt in enumerate(method_names):
    ax.annotate(txt, (max_hours_audio[i], relative_accuracy[i]))

plt.xlabel("Hours of Audio")
plt.ylabel("Relative Accuracy")
plt.title("Relative Accuracy by Hours of Audio at 10 Wh Budget (SR = 16000)")
plt.grid()
plt.show()


wattseconds_per_second_audio = np.array(ys)

kwh_per_hour_audio = wattseconds_per_second_audio * 3600 * 2.7778E-7

print(kwh_per_hour_audio)

plt.figure()
plt.barh(x_pos, kwh_per_hour_audio, color='green')
plt.ylabel("Method")
plt.xlabel("kWh")
plt.title("kWh Required to Infer One Hour of Audio")

plt.yticks(x_pos, method_names)

plt.show()


carbon_intensity = 1.0677

plt.figure()
plt.barh(x_pos, kwh_per_hour_audio * carbon_intensity, color='green')
plt.ylabel("Method")
plt.xlabel("Lbs of C02")
plt.title("Lbs of C02 Emitted to Infer One Hour of Audio")

plt.yticks(x_pos, method_names)

plt.show()

