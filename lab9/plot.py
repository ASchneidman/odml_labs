import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import librosa

POWER_FACTOR = 7.2

#audio_path = os.path.join('..', os.path.join('voxconverse', 'audio'))
audio_path = '../lab7'

with open('interview_stats.pkl', 'rb') as f:
    stats = pickle.load(f)

sample_rates = sorted(set([s[1] for s in stats.keys()]))
method_names = set([s[0] for s in stats])

# plot power
plt.figure()
for name in method_names:
    y_values = []
    #errors = []
    for sample_rate in sample_rates:
        power_values = np.array([d['time'] for d in stats[(name, sample_rate)].values()]).flatten() * POWER_FACTOR

        y_values.append(power_values.sum())

    plt.plot(sample_rates, y_values, label=name)

plt.legend()
plt.title('Total Power (Watt Seconds) over Sampling Rate')
plt.xlabel('Sampling Rate')
plt.ylabel('Watt Seconds')
plt.show()

# plot time
plt.figure()
for name in method_names:
    y_values = []
    errors = []
    for sample_rate in sample_rates:
        normalized_inference_times = []
        for file_name in stats[(name, sample_rate)]:
            duration = librosa.get_duration(filename=os.path.join(audio_path, f"{file_name}"))
            for t in stats[(name, sample_rate)][file_name]['time']:
                normalized_inference_times.append(t / duration)


        normalized_inference_times = np.array(normalized_inference_times)

        y_values.append(normalized_inference_times.mean())
        errors.append(normalized_inference_times.std())

    plt.errorbar(sample_rates, y_values, yerr=errors, label=name)

plt.legend()
plt.title('Normalized Inference Time over Sampling Rate')
plt.ylabel('Normalized Inference Time (Extra processing time / audio length)')
plt.xlabel('Sampling Rate')
plt.show()


# plot relative accuracy
plt.figure()
for name in method_names:
    y_values = []
    errors = []
    for sample_rate in sample_rates:
        acc = np.array([d['accuracy'] for d in stats[(name, sample_rate)].values()]).flatten()

        y_values.append(acc.mean())
        errors.append(acc.std())

    plt.errorbar(sample_rates, y_values, yerr=errors, label=name)

plt.legend()
plt.title('Relative Accuracy over Sampling Rate')
plt.xlabel('Sampling Rate')
plt.ylabel('Relative Accuracy')
plt.show()
