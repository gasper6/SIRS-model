# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 14:55:53 2019

@author: Janko
"""

print('Importing packages')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
#import plotly.graph_objs as go
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

## Time parameters settings
print('Setting model parameters...')
print('Setting time parameters...')
dt     =  .001;       # time step
tmax   =  20;        # time horizon for t (in years)
taumax =  20;       # time horizon for tau (in years)
ximax  =  taumax;    # time horizon for xi (in years)
# taumax, ximax should be large enough that PR, PV < 1e-8

t   = np.arange(0, 1+tmax/dt)   * dt     # time discretization for t
tau = np.arange(0, 1+taumax/dt) * dt    # time discretization for tau
xi  = np.arange(0, 1+ximax/dt)  * dt     # time discretization for xi

nt   = len(t);   # number of steps in t,   used in memory allocation
ntau = len(tau); # number of steps in tau
nxi  = len(xi);  # number of steps in xi

# population and disease parameter setting
print('Setting population and disease parameters...')
beta = 50  #250;#140;#4.123;
x = 0.2    #0.95;      # vaccintaion rate
mu = .02       # birth date
muM = mu       # mortality rate
d = 10/365      # avg_years_of_infection
gamma = 1/d # 365/avg_days_of_infection

# pre-calculated prob. of not dying for every time step
# useful for vectorized filling of diagonals V and R
mort = (1 - muM*dt)**(np.arange(0, max(nt, ntau, nxi)));
# mort = exp(-mu * dt * (0:max([nt, ntau, nxi]))); # exponential version

N0 = 1e5      # initial population size
I0 = 1        # initial number of infected individuals

# imunity waning setting
GMT0 = 1914
w_tau = 0.069 * 5
w_xi  = w_tau
sigma = 0.92   
Ccrit = 150

# SVF_T = normcdf((log(Ccrit/GMT0) + w_tau*tau)/sigma); # SVF(tau)
# SVF_X = normcdf((log(Ccrit/GMT0) + w_xi*xi) /sigma); # SVF(xi)
print('Creating SVF functions')
PR = norm.cdf((np.log(GMT0 / Ccrit) - w_tau*tau)/sigma) # P_R
PV = norm.cdf((np.log(GMT0 / Ccrit) - w_xi *xi) /sigma) # P_V

PRprime = norm.pdf((np.log(GMT0 / Ccrit) - w_tau*tau)/sigma) * (-w_tau/sigma) # PR'
PVprime = norm.pdf((np.log(GMT0 / Ccrit) - w_xi *xi) /sigma) * (-w_tau/sigma) # PV'

PPR = PRprime / PR
PPV = PVprime / PV

# SIRS-like profiles of PR, PV
# d_R = 20;             # Average years of lasting imunity
# theta = 1/d_R; clear d_R;
# PR = exp(-theta*tau); PV = exp(-theta*xi); clear theta;

# Collect garbage
del GMT0, w_tau, w_xi, sigma, Ccrit, d;

print('Interesting numbers. It would be nice if the first would be greater')
print('%.4f'%(1 - mu*x*(np.sum(PV)*dt)))
print('%.4f'%((gamma + muM)/beta))


## Memory allocation
print('Allocating memory...')

# First  coordinate is time
# Second coordinate is tau, xi

S = np.zeros((nt, 1))
I = np.zeros((nt, 1))
R = np.zeros((nt, ntau)) # R tilde
V = np.zeros((nt, nxi))  # V tilde
N = np.zeros((nt, 1))


## Initial conditions
print('Applying initial conditions')
N[0] = N0
I[0] = I0


# Considering constant birth rate and imunity waning, this should be the
# profile of vaccinated:
V[0,:] = mu*x*N0*PV*mort[0:nxi]

# For now, zero boundary for R is used:
# long time ago there were no recovered.
R[0,:] = 0

# Susceptible is everybody who is not in other compartments
S[0] = N0 - np.sum(V[0,:], 0)*dt - np.sum(R[0,:])*dt - I[0]
# Q: Is there a more math way to get S(1)? # Actually S(0). Thanks, matlab.
# Like all vaccinated who lost imunity and are still alive + all who were
# not vaccinated
# A: There is a way, but we do not want to do it now. We would need to know
# function P'/P.


# Once we know V(1,:), we shold be able to fill all upeer triangle
ind = np.arange(min(nt, nxi))
for i in range(nxi):
    if i%(np.ceil(nxi/20)) == 0:
        print('.', end='')
    n = min(nt, nxi-i)
    V[ind[:n], ind[:n]+i] = V[0,i]/PV[i]*PV[i:i+n]*mort[:n]
print()

# something similar for R(.,.) once we know boundary condition
ind = np.arange(min(nt, ntau))
for i in range(ntau):
    if i%(np.ceil(ntau/20)) == 0:
        print('.', end='')
    n = min(nt, ntau-i)
    R[ind[:n], ind[:n]+i] = R[0,i]/PR[i]*PR[i:i+n]*mort[:n]
print()

# Collect garbage
del n, ind


## Time loop
print('Starting time loop')
ind = np.arange(max(min(nt, ntau), min(nt, nxi)))
for i in range(nt-1):
    if i%200 == 0:
        print('%d/%d'%(t[i], tmax))
    I[i+1] = I[i] + beta*I[i]*S[i]/N[i]*dt - I[i]*gamma*dt - I[i]*muM*dt
    
    i += 1
    n = min(nt-i, ntau)
    R[ind[:n]+i, ind[:n]] = gamma*I[i-1] *PR[:n]*mort[:n]
    n = min(nt-i, nxi)
    V[ind[:n]+i, ind[:n]] = N[i-1]*mu*x*PV[:n]*mort[:n]
    
    i -= 1
    
    S[i+1] = S[i] + mu*N[i]*(1-x)*dt + mu*N[i]*x*(1-PV[0])*dt -\
             beta*S[i]*I[i]/N[i]*dt - S[i]*muM*dt + (1-PR[0])*gamma*I[i]*dt +\
             (mort[1]*np.sum(R[i,:]) - np.sum(R[i+1,1:])) * dt +\
             (mort[1]*np.sum(V[i,:]) - np.sum(V[i+1,1:])) * dt
    
    N[i+1] = S[i+1] + I[i+1] + np.sum(R[i+1, :])*dt + np.sum(V[i+1,:])*dt


## Plotting data
print('Plotting data...')


plt.plot(t, N)
plt.title('Total population')


plt.matshow(V[:, :nt].T, origin='lower', cmap=cm.jet)
plt.title('V(t, $\\xi$)')
plt.xlabel('t')
plt.ylabel('$\\xi$')
plt.gca().xaxis.tick_bottom()

plt.figure()
plt.matshow(R[:nt, :nt].T, origin='lower', cmap=cm.jet)
plt.xlabel('t')
plt.ylabel('$\\tau$')
plt.title('R(t, $\\tau$)')
plt.gca().xaxis.tick_bottom()

X, Y = np.meshgrid(t, tau[:nt])
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, R[:nt, :nt].T, cmap=cm.jet, rstride=100, cstride=100, alpha=1)
ax.plot(t, np.zeros(nt),  R[:nt, 0], color='k')
#plt.title('R(t,$\\tau$)')
plt.xlabel('t')
plt.ylabel('$\\tau$')
plt.tight_layout()
del X, Y

plt.figure()
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infectious')
plt.plot(t, np.sum(R, 1)*dt, 'g', label='Recovered')
plt.plot(t, np.sum(V, 1)*dt, 'k', label='Vaccinated')
plt.xlabel('t')
plt.legend()
plt.grid()
plt.tight_layout()


plt.figure()
plt.plot(t, S, 'b')
plt.title('Susceptible')
plt.xlabel('t')
plt.grid()
plt.tight_layout()


plt.figure()
plt.plot(t, I, 'r')
plt.title('Infectious')
plt.grid()
plt.tight_layout()

plt.figure()
plt.plot(t, np.sum(R,1)*dt, 'g')
plt.title('Recovered')
plt.grid()
plt.tight_layout()

plt.figure()
plt.plot(t, np.sum(V,1)*dt, 'k')
plt.title('Vaccinated')
plt.grid()
plt.tight_layout()

plt.figure()
plt.plot(S, I)
plt.grid()
ind = np.array([0, np.argmax(I), 1000+np.argmax(I[1000:])])
plt.scatter(S[ind], I[ind])
#plt.annotate("t=0", (93000, 300))
#plt.annotate("t=%.2f"%t[ind[1]], (72000, 2650))
#plt.annotate("t=%.2f"%t[ind[2]], (72000, 2100))
plt.title('S-I interaction')
plt.tight_layout()

