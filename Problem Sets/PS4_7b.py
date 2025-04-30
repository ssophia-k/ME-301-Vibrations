import matplotlib.pyplot as plt
import numpy as np

m = 2 #kg
k = 100 #N/m
c = 4 #kg/s

w = np.linspace(0, 20, 1000) #rad/s

H_real = (k - m * w**2) / ((k - m * w**2)**2 + (c * w)**2)
H_imag = (c * w) / ((k - m * w**2)**2 + (c * w)**2)

H_magnitude = np.sqrt(H_real**2 + H_imag**2)
H_phase = np.arctan2(H_imag, H_real)

plt.figure()

# First subplot: Real and Imaginary Parts
plt.subplot(3, 1, 1)
plt.plot(w, H_real, label='Real Part')
plt.plot(w, H_imag, label='Imaginary Part')
plt.title('Real and Imaginary Parts of H(w)')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('H(w)')
plt.legend()
plt.grid()

# Second subplot: Magnitude
plt.subplot(3, 1, 2)
plt.plot(w, H_magnitude, label='Magnitude')
plt.title('Magnitude of H(w)')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Magnitude')
plt.legend()
plt.grid()

# Third subplot: Phase
plt.subplot(3, 1, 3)
plt.plot(w, H_phase, label='Phase')
plt.title('Phase of H(w)')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Phase')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

