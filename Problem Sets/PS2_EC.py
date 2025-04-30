import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *  

class Integrator:
    """Integrator for a system of first-order ordinary differential equations
    of the form \dot x = f(t, x, u).
    """
    def __init__(self, dt, f):
        self.dt = dt
        self.f = f

    def step(self, t, x, u):
        raise NotImplementedError

#this is a good numerical integrator
class RK4(Integrator):
    def step(self, t, x, u):
        k1 = self.dt * self.f(t, x, u)
        k2 = self.dt * self.f(t+self.dt/2, x+k1/2, u)
        k3 = self.dt * self.f(t+self.dt/2, x+k2/2, u)
        k4 = self.dt * self.f(t+self.dt, x+k3, u)
        return x + (k1 + 2*k2 + 2*k3 + k4)/6


mu_s = 0.06
mu_k = 0.05
m = 250
theta = 20 #degrees
k = 3000
g=9.81

x_0 = 0.1
v_0 = 0.1


#Define all possible functions for our friction direction (depending on velocity direction)
#state space is x[0]=xdot, x[1]=x
def move_right(t, x, u):
    x_double_dot = -g*mu_k*np.cos(np.radians(theta)) - (k/m) * x[1]
    x_dot = x[0]
    return np.array([x_double_dot,x_dot])

def move_left(t, x, u):
    x_double_dot = g*mu_k*np.cos(np.radians(theta)) - (k/m) * x[1]
    x_dot = x[0]
    return np.array([x_double_dot,x_dot])

def zero_velocity(t, x, u):
    x_double_dot = - (k/m) * x[1]
    x_dot = x[0]
    return np.array([x_double_dot,x_dot])


t = 0
x = np.array([v_0, x_0])
u = 0
dt = 0.01; n = 50000

right_integrator = RK4(dt, move_right)
left_integrator = RK4(dt, move_left)
zero_integrator = RK4(dt, zero_velocity)

t_history = [0]
x_history = [x]


# Initialize a flag to control loop breaking
loop_continue = True

for i in range(n):
    if not loop_continue:
        break

    # check sign of our current velocity to decide which function to integrate next (since friction switches)
    #but because of numerical integration, i give it some wiggle room around zero (bc chances are we will never hit zero exactly)
    if (np.array(x_history)[-1, 0] > 0.00001):
        x = right_integrator.step(t, x, u)
    elif (np.array(x_history)[-1, 0] < -0.00001):
        x = left_integrator.step(t, x, u)
    else:
        # check if we overcome static friction whenever our velocity is zero
        if ( (np.array(x_history)[-1, 1] * k) >  (m * g * mu_s * np.cos(np.radians(theta))) ):
            x = zero_integrator.step(t, x, u)
        else:
            # Set flag to False to break the loop in the next iteration
            loop_continue = False
            continue  # Skip the rest of the current loop iteration

    t = (i + 1) * dt
    t_history.append(t)
    x_history.append(x)


plt.figure(1)
plt.plot(t_history, np.array(x_history)[:,1])
plt.xlabel('Time [s]')
plt.ylabel('Displacement x [m]')
plt.title("IC Response of Friction mass spring system")

plt.show()
