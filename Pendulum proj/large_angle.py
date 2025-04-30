import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *  
from scipy.integrate import solve_ivp

m = 0.0931 #mass of ruler, kg
L = 0.6096 #length of ruler, m
w = 0.03 #width of ruler, m
g = 9.81 #gravitational constant, m/s^2
I_com = (1/12) * m * (w**2 + L**2) #mass MoI about center of ruler, kg*m^2
I_pivot = I_com + m * (L/2)**2 #mass MoI shifted to the pivot point, kg*m^2

print(np.sqrt((m*g*L)/ (2*I_pivot)))

#Experimental natural frequency
omega_n = 4.779324275

#Initial conditions
theta_0 = np.radians(31.08040516)
omega_0 = 0


#Get time-step angles from tracker data
tracker_data = np.genfromtxt('ME-301 Vibrations/Large_angle_data.csv', delimiter=',', skip_header=1)
tracker_data = np.nan_to_num(tracker_data, nan=0.0)
real_time=(tracker_data[:,0]-tracker_data[0,0])
trial1 = tracker_data[:,1] 
trial2 = tracker_data[:,2] 
trial3 = tracker_data[:,3] 


#Set up experimental (using the natural frequency) solution
t = np.linspace(0, real_time[-1], len(real_time))
experimental = np.degrees(theta_0)*np.cos(omega_n*t)


#NON-LINEAR IC Model
def sys(t,x):
    theta, omega = x
    return [ omega, -((m*g*L)/ (2*I_pivot))*np.sin(theta)]
tspan = [t[0], t[-1]]

sol_nonlin = solve_ivp(sys, tspan, [theta_0,omega_0], t_eval=t, rtol=1e-6, atol=1e-9)
theta_nonlin = sol_nonlin.y[0,:]
omega_nonlin = sol_nonlin.y[1,:] 


#LINEARIZED Model
A = np.array([[0,1],
             [-((m*g*L)/(2*I_pivot)), 0] ])

B = np.array([[0],
             [0] ])

C = np.eye(2)

D = np.zeros((2,1))

sys_lin = ss(A, B, C, D)

# Linear IC Response
IC =np.array([[theta_0],
             [omega_0]])

y_lin, t_lin = initial(sys_lin, t ,IC )
theta_lin = y_lin[:,0]
omega_lin = y_lin[:,1]

#Models' RMSD
lin_rmsd = np.sqrt(np.sum([(theta_lin[i]-experimental[i])**2 for i in range(len(theta_lin))]) / len(theta_lin))
nonlin_rmsd = np.sqrt(np.sum([(theta_nonlin[i]-experimental[i])**2 for i in range(len(theta_nonlin))]) / len(theta_nonlin))
print(lin_rmsd)
print(nonlin_rmsd)


#Plot all our data
plt.figure(1)
plt.plot(t, np.degrees(theta_nonlin))
plt.plot (t, np.degrees(theta_lin))
plt.plot(t, experimental)
plt.legend(["nonlinear", "linear", "experimental freq."])
plt.xlabel('Time [s]')
plt.ylabel('Angle [deg]')
plt.title("IC Response of Ruler Pendulum, Experimental vs model, Theta_0 = 31.08º")

plt.figure(2)
plt.plot(t, experimental)
plt.plot(t, (trial1+trial2+trial3)/3)
plt.legend(["experimental freq.", "real experimental data"])
plt.xlabel('Time [s]')
plt.ylabel('Angle [deg]')
plt.title("IC Response of Ruler Pendulum, Real behavior vs idealized, Theta_0 = 31.08º")

plt.show()


