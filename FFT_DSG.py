"""
@author: HuidobroMG

"""

# Import the modules
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scop
import matplotlib.animation as animation

# First approach to the model
dx = 0.1
xmax = 20
x = np.arange(-xmax, xmax+dx, dx)
Nx = len(x)

# Parameters of the model, for R > arcsinh(1) there is a local minimum in the potential
R = 4.0
n = 0
T2 = np.tanh(R)**2
C = np.cosh(R)
C2 = C**2
phi_vac = -2*np.pi + 4*np.pi*n

# Potential
phi = np.linspace(0, 50, 200)
V = np.tanh(R)**2*(1-np.cos(phi)) + 4/np.cosh(R)**2*(1 + np.cos(phi/2))

# Kink and Antikink solutions
def Kink(x, R, n, anti):
    if anti == True:
        return 4*np.pi*n - 4*np.arctan(np.sinh(x)/np.cosh(R))
    else:
        return 4*np.pi*n + 4*np.arctan(np.sinh(x)/np.cosh(R))

phi_K = Kink(x, R, n, False)
phi_AK = Kink(x, R, n, True)

# Plot the solutions
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.set_title('Potential')
ax1.plot(phi, V, 'b-')

ax2.set_title('Kink-antikink')
ax2.plot(x, phi_K, 'b-')
ax2.plot(x, phi_AK, 'b-')

# Collisions

# Parameters of the spacetime grid
dx = 0.1
dt = 0.01
xmax = 30
x = np.arange(-xmax, xmax+dx, dx)
Nx = len(x)

ratio = dt/dx

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

# Initial configuration
a0 = 10
v0 = -0.2
gamma = 1/np.sqrt(1-v0**2)
phi = 4*(np.arctan(np.sinh(gamma*(x+a0))/C) - 
         np.arctan(np.sinh(gamma*(x-a0))/C)) + phi_vac
dphi_dt = 4*gamma*v0/C*(np.cosh(gamma*(x+a0))/(1+(np.sinh(gamma*(x+a0))/C)**2) + 
                      np.cosh(gamma*(x-a0))/(1+(np.sinh(gamma*(x-a0))/C)**2))

# Time grid
tmax = abs(2*a0/v0)
t = np.arange(0, tmax, dt)
Nt = len(t)

# Configuration in i = -1
phi_old = phi - dphi_dt*dt

# Fit at some time with the simple boosted kink function
def Lorentz_Fit(x, *pars):
    return 4*(np.arctan(np.sinh(pars[0]*(x+pars[1]))/C) - np.arctan(np.sinh(pars[0]*(x-pars[1]))/C)) + phi_vac

# Evolution
phi_t = []
phi_t.append(phi)
time = []
time.append(t[0])

gamma_vals = [gamma]
a_vals = [a0]

gamma_val = gamma
a_val = a0
counter = 0
for i in range(Nt):
    phi_new = EL(phi, phi_old)
    
    phi_old = 1.0*phi
    phi = 1.0*phi_new
    
    counter += 1
    if counter%20 == 0:
        time.append(t[i]+dt)
        phi_t.append(phi_new)
        gamma_val, a_val = scop.curve_fit(Lorentz_Fit, x, phi_new, p0 = np.array([gamma_val, a_val]))[0]
        gamma_vals.append(gamma_val)
        a_vals.append(a_val)

phi_t = np.array(phi_t)

# Animate the solution
fig = plt.figure()
ax = plt.axes(xlim = (-xmax, xmax), ylim = (-1.5*np.min(phi_t), 1.5*np.min(phi_t)))

line, = ax.plot([], [])
def update(i):
    line.set_data(x, phi_t[i])
    return line,

ani = animation.FuncAnimation(fig = fig, func = update, 
                              frames = len(time), interval = dt, blit = True)

#ani.save('FFT_DSG.gif', writer = 'pillow')

plt.show()





