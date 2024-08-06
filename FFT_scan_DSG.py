"""
@author: HuidobroMG

"""

# Import the modules
import numpy as np
import matplotlib.pyplot as plt

# Parameters of the spacetime grid
dx = 0.1
dt = 0.01
xmax = 100
tmax = 1400
x = np.arange(-xmax, xmax+dx, dx)
t = np.arange(0, tmax+dt, dt)
Nx = len(x)
Nt = len(t)

ratio = dt/dx

# Parameters of the model
R = 3.0
n = 0
T2 = np.tanh(R)**2
C = np.cosh(R)
C2 = C**2
phi_vac = -2*np.pi + 4*np.pi*n

# V and dV/dphi
def V(phi):
    return T2*(1-np.cos(phi)) + 4/C2*(1+np.cos(phi/2))

def dV(phi):
    return T2*np.sin(phi) - 2/C2*np.sin(phi/2)

# Matrix method
M = np.zeros((Nx, Nx))
for i in range(1, Nx-1):
    M[i, i-1] = ratio**2
    M[i, i] = 2*(1-ratio**2)
    M[i, i+1] = ratio**2

def EL(phi, phi_old):
    phi_new = np.dot(M, phi)
    phi_new -= dt**2*dV(phi) + phi_old
    # Impose boundary conditions
    phi_new[0] = phi_new[-1] = phi_vac
    return phi_new


# Evolution
def evolve(a0, v0):
    gamma = 1/np.sqrt(1-v0**2)
    phi = 4*(np.arctan(np.sinh(gamma*(x+a0))/C) - np.arctan(np.sinh(gamma*(x-a0))/C)) + phi_vac
    dphi_dt = 4*gamma*v0/C*(np.cosh(gamma*(x+a0))/(1+np.sinh(gamma*(x+a0))**2/C2) + np.cosh(gamma*(x-a0))/(1+np.sinh(gamma*(x-a0))**2/C2))

    # Configuration in i = -1
    phi_old = phi - dphi_dt*dt
    
    # Evolution
    phi_t = []
    time = []
    phi_t.append(phi)
    time.append(t[0])
    
    counter = 0
    for i in range(Nt):
        phi_new = EL(phi, phi_old)
        
        phi_old = 1.0*phi
        phi = 1.0*phi_new
        
        counter += 1
        if counter%20 == 0:
            time.append(t[i]+dt)
            phi_t.append(phi_new)

    return phi_t, time

# Scan the velocities space
a0 = 10
v = -np.arange(0.005, 0.035, 0.0001)

phi = []
phi_x0 = []
for v0_val in v:
    print('v0 = ', v0_val)
    phi_t, time = evolve(a0, v0_val)
    phi.append(phi_t)
    phi_x0.append(phi_t[Nx//2])

Ntime = len(time)
Nv = len(v)

# Create the contour figure
fig, ax = plt.subplots(figsize = (8, 5))
contourf_ = ax.contourf(np.reshape(np.repeat(np.abs(v), Ntime), (Nv, Ntime)),
                        time, phi_x0, vmin = np.min(phi_x0),
                        vmax = np.max(phi_x0), levels = 60)

cbar = fig.colorbar(contourf_)
cbar.ax.set_title(r'$\phi (0, t)$', fontsize = 15)

ax.set_xlabel(r'$v$', fontsize = 20)
ax.set_ylabel(r'$t$', fontsize = 20)

plt.show()
