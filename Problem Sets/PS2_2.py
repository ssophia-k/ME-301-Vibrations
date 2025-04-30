import numpy as np
import matplotlib.pyplot as plt

m = 100 #kg
k = 3000 #N/m
b = 300 #kg/s

w_n = np.sqrt(k/m)
zeta = (b/(2*m)) / w_n
w_d = w_n * np.sqrt(1-zeta**2)

A = 0.1 
B = (zeta * w_n * A) / w_d

t = np.linspace(0, 5, 1000)

x = np.exp(-zeta * w_n * t) * (A*np.cos(w_d * t) + B*np.sin(w_d * t))
die_out_line = 0.01 * A + (0*t)

plt.figure(1)
plt.plot(t, x)
plt.plot(t, die_out_line, linestyle='dashed')
intersection = np.argwhere(np.diff(np.sign(x - die_out_line))).flatten()
plt.plot(t[intersection], x[intersection], 'ro')
plt.xlabel('Time [s]')
plt.ylabel('Displacement x [m]')
plt.title("IC Response of Damped mass spring system")

plt.show()

print(t[intersection[-1]])