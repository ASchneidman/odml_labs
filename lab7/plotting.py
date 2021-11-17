import numpy as np
import matplotlib.pyplot as plt

X = np.load('./metrics.npz')['arr_0']
nonsecure_inf_time = np.load('./metrics.npz')['arr_1']
secure_inf_time = np.load('./metrics.npz')['arr_2']
accuracies = np.load('./metrics.npz')['arr_3']


plt.figure
plt.plot(X, nonsecure_inf_time, label='Nonsecure Query')
plt.plot(X, secure_inf_time, label='Secure Query with Biometric Salting')
plt.title('Query Times vs. Sampling Rate for Interview Test Audio')
plt.ylabel('Query Times')
plt.xlabel('Audio Sampling Rate')
plt.grid()
plt.legend()
plt.savefig("inference_times.png")

plt.figure
plt.plot(X, accuracies, label='Diarization Accuracy')
plt.title('Inference Times vs. Sampling Rate for Interview Test Audio')
plt.ylabel('Inference Times')
plt.xlabel('Input Size')
plt.grid()
plt.legend()
plt.savefig("accuracies.png")

