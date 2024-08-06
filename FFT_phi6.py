"""
@author: HuidobroMG


"""

# Import the modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters of the spacetime grid
dx = 0.1
dt = 0.01
xmax = 30
x = np.arange(-xmax, xmax+dx, dx)
Nx = len(x)

ratio = dt/dx

# phi6 model, V = phi**2*(1-phi**2)**2
phiK_1 = np.sqrt((1 + np.tanh(x))/2) # Kink which joins the 0 and 1 vacua
phiAK_1 = np.sqrt((1 - np.tanh(x))/2) # Antikink which joins the 1 and 0 vacua
phiK_2 = -np.sqrt((1 - np.tanh(x))/2) # Kink which joins the -1 and 0 vacua
phiAK_2 = -np.sqrt((1 + np.tanh(x))/2) # Antikink which joins the 0 and -1 vacua

# Matrix method
M = np.zeros((Nx, Nx))
for i in range(1, Nx-1):
    M[i, i-1] = ratio**2
    M[i, i] = 2*(1-ratio**2)
    M[i, i+1] = ratio**2

def EL(phi, phi_old):
    dVpot = -2*phi*(1-phi**2)
    phi_new = np.dot(M, phi)
    phi_new -= dt**2*dVpot + phi_old
    # Impose boundary conditions
    phi_new[0] = phi_new[-1] = -1
    return phi_new


# Initial configuration
a0 = 10
v0 = -0.5
gamma = 1/np.sqrt(1-v0**2)
phi = np.tanh(gamma*(x+a0)) - np.tanh(gamma*(x-a0)) - 1
dphi_dt = gamma*v0*(1/np.cosh(gamma*(x+a0))**2 + 1/np.cosh(gamma*(x-a0))**2)

# Time grid
tmax = abs(2*a0/v0)
t = np.arange(0, tmax, dt)
Nt = len(t)

# Configuration in: i = -1
phi_old = phi - dphi_dt*dt

# Evolution
phi_t = []
phi_t.append(phi)
time = []
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


# Animate the solution
fig = plt.figure()
ax = plt.axes(xlim = (-xmax, xmax), ylim = (-2, 1.5))

line, = ax.plot([], [])
def update(i):
    line.set_data(x, phi_t[i])
    return line,

# Animate
ani = animation.FuncAnimation(fig = fig, func = update, 
                              frames = len(time), interval = dt, blit = True)

#ani.save('FFT_phi6.gif', writer = 'pillow')

plt.show()