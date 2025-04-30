import matplotlib.pyplot as plt
import numpy as np

F0 = 10 #N
w = 1 #rad/s

m = 1 #kg
c = 4 #kg/s
k = 9 #N/m


# PART A
t = np.linspace(0, 30, 1000) #s
M = 1/ np.sqrt((k - m * w**2)**2 + (c * w)**2)
Phi = np.arctan2(c * w, k - m * w**2)
x_p = F0 * M * np.sin(w * t - Phi) 

plt.figure()
plt.plot(t, x_p)
plt.title('Forced solution of the system over time')
plt.xlabel('Time [s]')
plt.ylabel('Displacement [m]')
plt.grid()
plt.show()


# PART B
t1 = np.linspace(0, 4, 1000) #before impulse
t2 = np.linspace(4, 30, 1000) #after impulse

F_d = 10 #N

w_n = np.sqrt(k/m) #natural frequency of the system
zeta = c/(2*m*w_n) #damping ratio of the system 
w_d = w_n * np.sqrt(1 - zeta**2) #damped frequency of the system

A = 0.448
B = 0.537

x_p1 = F0 * M * np.sin(w * t1 - Phi) 
x_p2 = F0 * M * np.sin(w * t2 - Phi)
x_d = (F_d / (m * w_d) ) * np.exp(-zeta * w_n * (t2-4)) * np.sin(w_d * (t2-4))
x_h1 = np.exp(-zeta * w_n * t1) * (A * np.sin(w_d * t1) + B * np.cos(w_d * t1))
x_h2 = np.exp(-zeta * w_n * t2) * (A * np.sin(w_d * t2) + B * np.cos(w_d * t2))

plt.figure()
plt.plot(t1, x_h1 + x_p1)
plt.plot(t2, x_h2 + x_p2 + x_d)
plt.title('Forced + impulse + IC response of the system over time')
plt.xlabel('Time [s]')
plt.ylabel('Displacement [m]')
plt.grid()
plt.show()
