# -*- coding: utf-8 -*-
"""
@author: HuidobroMG

Description:
    
    In this code I am solving the full field theory equation of motion
    numerically for the collision of a kink and an antikink 
    in the 1-dimensional phi6 theory.

"""

#------------------------------------------------------------------------------

# Packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#------------------------------------------------------------------------------

# Parameters of the grid
dx = 0.1
dt = 0.01
xmax = 30
x = np.arange(-xmax, xmax, dx)
Nx = len(x)

ratio = dt/dx

# phi6 model: V = phi**2*(1-phi**2)**2
phiK_1 = np.sqrt((1 + np.tanh(x))/2) # Kink which joins the 0 and 1 vacua
phiAK_1 = np.sqrt((1 - np.tanh(x))/2) # Antikink which joins the 1 and 0 vacua
phiK_2 = -np.sqrt((1 - np.tanh(x))/2) # Kink which joins the -1 and 0 vacua
phiAK_2 = -np.sqrt((1 + np.tanh(x))/2) # Antikink which joins the 0 and -1 vacua

#------------------------------------------------------------------------------

# Define the matrix of the system
M = np.zeros((Nx, Nx))
for i in range(1, Nx-1):
    M[i, i-1] = ratio**2
    M[i, i] = 2*(1-ratio**2)
    M[i, i+1] = ratio**2

# Updating the field function
def EL(phi, phi_old):
    dVpot = -2*phi*(1-phi**2)
    phi_new = np.dot(M, phi)
    phi_new -= dt**2*dVpot + phi_old
    
    phi_new[0] = phi_new[-1] = -1
    
    return phi_new


# Initial configuration
a0 = 10
v0 = -0.1
gamma = 1/np.sqrt(1-v0**2)

tmax = abs(2*a0/v0)
t = np.arange(0, tmax, dt)
Nt = len(t)

phi = np.tanh(gamma*(x+a0)) - np.tanh(gamma*(x-a0)) - 1
dphi_dt = gamma*v0*(1/np.cosh(gamma*(x+a0))**2 + 1/np.cosh(gamma*(x-a0))**2)

# Configuration in: i = -1
phi_old = phi - dphi_dt*dt

#------------------------------------------------------------------------------

phi_t = []
phi_t.append(phi)
time = []
time.append(t[0])

# Evolution of the system in time
counter = 0
for i in range(Nt):
    phi_new = EL(phi, phi_old)
    
    phi_old = 1.0*phi
    phi = 1.0*phi_new
    
    counter += 1
    if counter%20 == 0:
        time.append(t[i]+dt)
        phi_t.append(phi_new)


#------------------------------------------------------------------------------

# Animate the solution in time and create a GIF of the collision
Animer = 0

if Animer == 1:

    fig = plt.figure()
    ax = plt.axes(xlim=(-xmax, xmax), ylim=(-2, 1.5))
    line, = ax.plot([], [], lw=3)

    def init():
        line.set_data([], [])
        return line,
    def animate(i):
        y = phi_t[i]
        line.set_data(x, y)
        ax.set_title('t = {}'.format(np.round(time[i], 2)))
        return line,

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames = len(time), interval = 1e4/len(time),
                         blit = True)

    anim.save('FFT_KAK_phi6.gif', writer = 'pillow')