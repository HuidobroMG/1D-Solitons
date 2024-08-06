"""
@author: HuidobroMG

"""

# Import the modules
import numpy as np
import matplotlib.pyplot as plt

# Parameters of the spacetime grid
dx = 0.1
dt = 0.005
xmax = 50
tmax = 850
x = np.arange(-xmax, xmax+dx, dx)
t = np.arange(0, tmax+dt, dt)
Nx = len(x)
Nt = len(t)

ratio = dt/dx

# Energy of the system
def Energy(phi, phi_old):
    dphi_dx = np.zeros(Nx)
    dphi_dt = (phi-phi_old)/dt
    for i in range(1, Nx-1):
        dphi_dx[i] = (phi[i+1]-phi[i])/dx
    
    Upot = phi**2*(1-phi**2)**2/2
    E = np.trapz(x, 0.5*dphi_dt**2 + 0.5*dphi_dx**2 + Upot)
    return E


# Matrix method
M = np.zeros((Nx, Nx))
for i in range(1, Nx-1):
    M[i, i-1] = ratio**2
    M[i, i] = 2*(1-ratio**2)
    M[i, i+1] = ratio**2


def EL(phi, phi_old):
    dVpot = phi*(1-phi**2)*(1-3*phi**2)
    phi_new = np.dot(M, phi)
    phi_new -= dt**2*dVpot + phi_old
    # Impose boundary conditions
    phi_new[0] = phi_new[-1] = 1
    return phi_new

# Time-evolution function
def evolve(vinic):
    a0, v0 = vinic
    gamma = 1/np.sqrt(1-v0**2)
    phi = np.sqrt((1-np.tanh(gamma*(x+a0)))/2) + np.sqrt((1+np.tanh(gamma*(x-a0)))/2)
    dphi_dt = -gamma*v0/(2*np.sqrt(2))*((1-np.tanh(gamma*(x-a0)))*np.sqrt(1+np.tanh(gamma*(x-a0))) + (1+np.tanh(gamma*(x+a0)))*np.sqrt(1-np.tanh(gamma*(x+a0))))
    
    # Configuration in i = -1
    phi_old = phi - dphi_dt*dt
    
    phi_t = []
    phi_t.append(phi)
    time = []
    time.append(t[0])
    
    E = []
    E.append(Energy(phi, phi_old))
    
    # Evolution
    counter = 0
    for i in range(Nt):
        phi_new = EL(phi, phi_old)
        
        phi_old = 1.0*phi
        phi = 1.0*phi_new
        
        counter += 1
        if counter%20 == 0:
            time.append(t[i]+dt)
            phi_t.append(phi_new)
            E.append(Energy(phi, phi_old))
    
    # Error in the conservation of E
    Emax = np.max(E)
    Emin = np.min(E)
    print('Error (%)', abs((Emax-Emin)/Emin)*100)
    return phi_t, time

# Scan the velocities space
a0 = 10
v0 = np.arange(0.01, 0.05, 0.001)
phi = np.zeros((len(v0), Nt//20+1))
for i in range(len(v0)):
    print(i)
    phi_t, time = evolve([a0, -v0[i]])
    phi_t = np.transpose(phi_t)
    phi[i] = np.flip(phi_t[:][Nx//2])
    print('-----------------------')

# Create the contour figure
fig, ax = plt.subplots(figsize = (8, 5))

contourf_ = ax.contourf(v0, np.flip(time), np.transpose(phi), vmin = 0, vmax = np.max(phi), levels = 10)

cbar = fig.colorbar(contourf_)
cbar.ax.set_title(r'$\phi (0, t)$', fontsize = 15)

ax.set_xlabel(r'$v$', fontsize = 20)
ax.set_ylabel(r'$t$', fontsize = 20)

plt.show()

