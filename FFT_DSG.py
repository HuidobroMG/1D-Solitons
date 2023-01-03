# -*- coding: utf-8 -*-
"""
@author: HuidobroMG

Description:
    
    In this code I am solving the full field theory equation of motion
    numerically for the collision of a kink and an antikink 
    in the 1-dimensional double sine-Gordon theory.

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
xmax = 50
x = np.arange(-xmax, xmax+dx, dx)
Nx = len(x)

r = dt/dx

# Parameters of the model
R = 1.5
T2 = np.tanh(R)**2
C = np.cosh(R)
C2 = C**2
phi_vac = -2*np.pi # Vacuum value of the DSG potential

# V and dV/dphi.
def V(phi):
    return T2*(1-np.cos(phi)) + 4/C2*(1+np.cos(phi/2))

def dV(phi):
    return T2*np.sin(phi) - 2/C2*np.sin(phi/2)


# Differential equation
M = np.zeros((Nx, Nx))
for i in range(1, Nx-1):
    M[i, i-1] = r**2
    M[i, i] = 2*(1-r**2)
    M[i, i+1] = r**2

def EL(phi, phi_old):
    
    phi_new = np.dot(M, phi)
    phi_new -= dt**2*dV(phi) + phi_old
    
    phi_new[0] = phi_new[-1] = phi_vac
    
    return phi_new

#------------------------------------------------------------------------------

# Initial configuration
a0 = 10
v0 = -0.005
gamma = 1/np.sqrt(1-v0**2)

tmax = abs(2*a0/v0)
t = np.arange(0, tmax, dt)
Nt = len(t)
interv = int(0.1*tmax)

# Kink-Antikink configuration
phi = 4*(np.arctan(np.sinh(gamma*(x+a0))/C) - 
         np.arctan(np.sinh(gamma*(x-a0))/C)) + phi_vac
dtphi = 4*gamma*v0/C*(np.cosh(gamma*(x+a0))/(1+np.sinh(gamma*(x+a0))**2/C2) + 
                      np.cosh(gamma*(x-a0))/(1+np.sinh(gamma*(x-a0))**2/C2))

# Configuration in i = -1
phi_old = phi - dtphi*dt

#------------------------------------------------------------------------------

# Time evolution
phi_t = []
time = []
phi_t.append(phi)
time.append(t[0])

counter = 0
for i in range(1, Nt):
    phi_new = EL(phi, phi_old)
    
    phi_old = 1.0*phi
    phi = 1.0*phi_new
    
    counter += 1
    if counter%interv == 0:
        time.append(t[i]+dt)
        phi_t.append(phi_new)

#------------------------------------------------------------------------------

# Animation of the solution
Animer = 1

if Animer == 1:

    fig = plt.figure()
    ax = plt.axes(xlim=(-2.5*a0, 2.5*a0), ylim=(1.2*np.min(phi_t), 1.2*np.max(phi_t)))
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

    anim.save('FFT_DSG_KAK.gif', writer = 'pillow')





