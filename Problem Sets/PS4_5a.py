import matplotlib.pyplot as plt
import numpy as np

F0 = 500 #N
tau = 0.1 #s

magnitude = np.zeros(4)
frequency = np.zeros(4)
for n in range(1, 5):
    frequency[n-1] = 2 * n * np.pi / tau
    magnitude[n-1] = F0 / n


# Plot the amplitude spectrum
plt.figure(figsize=(6,4))
plt.stem(frequency, magnitude)
plt.xlabel("Frequency [rad/s]")
plt.ylabel("Fourier sine coefficient [N]")
plt.title("Spectrum of f(t) (first 4 sine coefficients)")
plt.grid(True)
plt.show()

