import matplotlib.pyplot as plt
import numpy as np

t1 = np.linspace(0, np.pi, 1000) #before impulse
t2 = np.linspace(np.pi, 10, 1000) #after impulse

x_p = np.exp(-t2) * np.sin(t2-np.pi)

x_h1 = np.exp(-t1) * (np.sin(t1)+np.cos(t1))
x_h2 = np.exp(-t2) * (np.sin(t2)+np.cos(t2))

plt.figure()
plt.plot(t1, x_h1)
plt.plot(t2, x_h2 + x_p)
plt.title('Impulse + IC response of the system over time')
plt.xlabel('Time [s]')
plt.ylabel('Displacement [m]')
plt.grid()
plt.show()