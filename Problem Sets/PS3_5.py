import control.matlab as mt
import matplotlib.pyplot as plt
import numpy as np

m1 = 1000. #kg, car with one passenger
m4 = 1240. #kg, car with four passengers
c = 2000. #kg/s, damping constant
k = 10000. #N/m, spring constant
Y = 0.02 #m, road bump height

v_specifics = [20, 60, 80, 100, 140] #km/h, car velocities of interest

for velocity in v_specifics:
    w = ((2. * np.pi) / 6.) * velocity * (1000. / 3600.)
    M1 = np.sqrt( (k**2 + (c*w)**2) / ( (c*w)**2 + (k - m1 * w**2)**2 ) ) * Y
    M4 = np.sqrt( (k**2 + (c*w)**2) / ( (c*w)**2 + (k - m4 * w**2)**2 ) ) * Y
    print("At velocity %d km/h, the displacement of the car with one passenger is %.5f m and with four passengers is %.5f m" % (velocity, M1, M4))


#Now do the plotting
v = np.linspace(0, 140, 1000)

w = ((2. * np.pi) / 6.) * v * (1000. / 3600.) #natural frequency of input in rad/s

M1 = np.sqrt( (k**2 + (c*w)**2) / ( (c*w)**2 + (k - m1 * w**2)**2 ) ) * Y
M4 = np.sqrt( (k**2 + (c*w)**2) / ( (c*w)**2 + (k - m4 * w**2)**2 ) ) * Y

plt.figure()
plt.plot(v, M1, label = 'One passenger')
plt.plot(v, M4, label = 'Four passengers')
plt.legend()
plt.xlabel('Driving speed [km/hr]'); plt.ylabel('Car displacement [m]'),
plt.title("Displacement of car versus driving speed for bumpy road modeled as cosine")
plt.show()

plt.figure()
plt.plot(w, M1, label = 'One passenger')
plt.plot(w, M4, label = 'Four passengers')
plt.legend()
plt.xlabel('Base excitation frequency [rad/s]'); plt.ylabel('Car displacement [m]'),
plt.title("Displacement of car versus input frequency of bumpy road modeled as cosine")
plt.show()
