import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# =========================
# 1. Charger le fichier
# =========================

audio, fs = sf.read("musique.wav")

print("Fréquence d'échantillonnage :", fs, "Hz")
print("Nombre d'échantillons :", len(audio))

# Si le fichier est stéréo, on prend le canal gauche
if audio.ndim == 2:
    signal = audio[:, 0]
else:
    signal = audio

# =========================
# 2. Domaine temporel
# =========================

N = len(signal)

# Vecteur temps
t = np.arange(N) / fs

plt.figure(figsize=(12, 5))

plt.plot(t, signal)

plt.title("Signal audio - Domaine temporel")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.show()

# =========================
# 3. Domaine fréquentiel
# =========================

# FFT
fft_signal = np.fft.rfft(signal)

# Fréquences correspondantes
frequencies = np.fft.rfftfreq(N, 1 / fs)

# Amplitude du spectre
amplitude = np.abs(fft_signal) / N

plt.figure(figsize=(12, 5))

plt.plot(frequencies, amplitude)

plt.title("Signal audio - Domaine fréquentiel")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.grid()

# On s'intéresse généralement aux fréquences audibles
plt.xlim(0, 20000)

plt.show()