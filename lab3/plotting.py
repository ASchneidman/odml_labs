
import numpy as np
import cv2
import matplotlib.pyplot as plt



X = np.load('./hubert.npz')['arr_0']
#y_hubert = np.load('./hubert.npz')['arr_1']
y_mod_cpc = np.load('./modified_cpc.npz')['arr_1']
y_vq_wav2vec = np.load('./vq_wav2vec.npz')['arr_1']
y_wav2vec = np.load('./wav2vec.npz')['arr_1']
#y_wav2vec2 = np.load('./wav2vec2.npz')['arr_1']
#y_wav2vec2_xlsr = np.load('./wav2vec2_xlsr.npz')['arr_1']


plt.figure
#plt.plot(X, y_hubert, label='HuBERT')
plt.plot(X, y_mod_cpc, label='Modified CPC')
plt.plot(X, y_vq_wav2vec, label='VQ Wav2Vec')
plt.plot(X, y_wav2vec, label='Wav2Vec')
#plt.plot(X, y_wav2vec2, label='Wav2Vec 2.0')
#plt.plot(X, y_wav2vec2_xlsr, label='Wav2Vec XLSR')

plt.title('Inference Times vs. Input Size for 3 Audio Embedding Models')
plt.ylabel('Inference Times')
plt.xlabel('Input Size')
plt.grid()
plt.legend()

plt.savefig("plot.png")