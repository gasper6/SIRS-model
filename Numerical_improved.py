# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 14:55:53 2019

@author: Janko
"""

print('Importing packages...')

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import argrelextrema
import os
import datetime 

save = True
saveformat = "pdf"

## Time parameters settings
print('Setting model parameters...')
print('Setting time parameters')
dt     =  .001      # time step
tmax   =  100       # time horizon for t (in years)
taumax =  100        # time horizon for tau (in years)
ximax  =  taumax    # time horizon for xi (in years)
# taumax, ximax should be large enough that PR, PV < 1e-8

tmin = -max(taumax, ximax)  # history
t0   = -int(tmin/dt)        # index of t, such that t[t0] = 0

# time discretization for t, tau, xi
t   = np.arange(1+int((tmax-tmin)/dt))*dt + tmin
tau = np.arange(1+int(taumax/dt))*dt
xi  = np.arange(1+int(ximax/dt))*dt

# number of steps in t from tmin to tmax, ntau, nxi
nt   = len(t)
ntau = len(tau)
nxi  = len(xi)

# population and disease parameter setting
print('Setting population and disease parameters')
beta = 50      # transmission rate
x = 0.2        # vaccintaion rate
mu = .02       # birth date
muM = mu       # mortality rate
d = 10/365     # avg_years_of_infection
gamma = 1/d

# pre-calculated prob. of not dying for t=0*dt, 1*dt, 2*dt, ...
# used for computing R and V
mort = (1 - muM*dt)**(np.arange(nt))
nat = mu*dt
mort  = np.exp(-muM * np.arange(nt)*dt)  # continuous eqivalent of muM*dt
nat   = 1 - np.exp(-mu * dt)
mort1 = 1 - mort[1]

N0 = 1e5      # initial population size
I0 = 1        # initial number of infected individuals

# imunity waning setting
GMT0  = 1914
w_tau = 0.069
w_xi  = w_tau
sigma = 0.92   
Ccrit = 150

#SVF_T = norm.cdf((np.log(Ccrit/GMT0) + w_tau*tau)/sigma); # SVF(tau)
#SVF_X = norm.cdf((np.log(Ccrit/GMT0) + w_xi*xi) /sigma); # SVF(xi)
print('Creating SVF functions...')
PR = norm.cdf((np.log(GMT0 / Ccrit) - w_tau*tau)/sigma) # P_R
PV = norm.cdf((np.log(GMT0 / Ccrit) - w_xi *xi) /sigma) # P_V

# Convolution P function, refined for deaths
PRmort = PR[::-1] * mort[-(nt-ntau+1)::-1]
PVmort = PV[::-1] * mort[-(nt-nxi +1)::-1]

    
print('time step: %.2g'%dt)
print('tmax: %.2g'%tmax)
print('taumax: %.2g'%taumax)
print('ximax: %.2g'%ximax)

print('beta: %.2g'%beta)
print('x: %.1g'%x)
print('mu: %.2g'%mu)
print('muM: %.2g'%muM)
print('gamma: %.2g'%gamma)
print('GMT0: %.3g'%GMT0)
print('w_tau: %.2g'%w_tau)
print('w_xi: %.2g'%w_xi)
print('sigma: %.2g'%sigma)
print('Ccrit: %.2g'%Ccrit)

if save is True:
    try:
        results_dir = str(datetime.datetime.now().timestamp())
        os.mkdir(results_dir)
        with open(results_dir + "/parameters.txt", 'w') as f:
            f.write(str(datetime.datetime.now())+'\n\n')
    
            f.write('time step: %.2g\n'%dt)
            f.write('tmax: %.2g\n'%tmax)
            f.write('taumax: %.2g\n'%taumax)
            f.write('ximax: %.2g\n\n'%ximax)
    
            f.write('beta: %.2g\n'%beta)
            f.write('x: %.1g\n'%x)
            f.write('nu: %.2g\n'%mu)
            f.write('mu: %.2g\n'%muM)
            f.write('gamma: %.2g\n'%gamma)
            f.write('GMT0: %.3g\n'%GMT0)
            f.write('w_tau: %.2g\n'%w_tau)
            f.write('w_xi: %.2g\n'%w_xi)
            f.write('sigma: %.2g\n'%sigma)
            f.write('Ccrit: %.2g\n\n'%Ccrit)
            
            f.write('min(PR): %.2g\n\n'%PR[-1])
            f.write('min(PV): %.2g\n\n'%PV[-1])
            
            f.write('1 - mu*x*(np.sum(PV)*dt):\n')
            f.write('%.4f\n'%(1 - mu*x*(np.sum(PV)*dt)))
            
            f.write('(gamma + muM)/beta:\n')
            f.write('%.4f\n'%((gamma + muM)/beta))
    
    except OSError:
        results_dir = None
else:
    results_dir = None

# Collect garbage
del GMT0, w_tau, w_xi, sigma, Ccrit, d


## Memory allocation
print('Allocating memory...')

S = np.zeros((nt,))
I = np.zeros((nt,))
R = np.zeros((nt,)) 
V = np.zeros((nt,))
N = np.zeros((nt,))


## Initial conditions
print('Applying initial conditions...')
N[:t0+1] = N0
I[t0] = I0


# With constant birth rate and population, imunity waning, this should be the
# profile of vaccinated:
V[:t0+1] = N0 * mu *x *(np.sum(PV[:] * mort[:nxi]) * dt)

# For now, zero boundary for R is used:
# long time ago there were no recovered.
R[:t0+1] = 0

# Susceptible is everybody who is not vaccinated, recovered nor infected
S[:t0+1] = N[:t0+1] - V[:t0+1] - R[:t0+1] - I[:t0+1]

print()
print('Interesting numbers. There should be outbreak if the first is greater.')
print('%.4f'%(1 - mu*x*(np.sum(PV)*dt)))
print('%.4f'%((gamma + muM)/beta))



print("Saving initial condition...")
if results_dir is not None:
    np.savetxt(results_dir+"/par_PR.txt",  PR, fmt='%g')
    np.savetxt(results_dir+"/par_PV.txt",  PV, fmt='%g')
    np.savetxt(results_dir+"/par_PRmort.txt",  PRmort, fmt='%g')
    np.savetxt(results_dir+"/par_PVmort.txt",  PVmort, fmt='%g')
    np.savetxt(results_dir+"/par_mort.txt",  mort, fmt='%g')

print_flag = 0  # just for printing progress (print every few sec)
## Time loop
print()
print('Starting time loop...')
print()
for i in range(t0, nt-1):
    if print_flag > 2**30:
        print('%.3g/%.3g'%(t[i], tmax))
        print_flag = 0
    I[i+1] = I[i] + I[i] * beta*S[i]/N[i]*dt - I[i] *gamma*dt - I[i] *mort1

    R[i+1] = gamma * np.inner(PRmort, I[(i-ntau+1):(i+1)]) * dt

    V[i+1] = mu*x* np.inner(PVmort,  N[(i-nxi+1):(i+1)]) * dt

    S[i+1] = S[i] + nat*N[i] - S[i]*mort1 -\
             beta*S[i]*I[i]/N[i]*dt + gamma*I[i]*dt +\
             (R[i]*mort[1] - R[i+1]) +\
             (V[i]*mort[1] - V[i+1])

#    S[i+1] = S[i] + nat*N[i]*(1-x) + nat*N[i]*x*(1-PV[0]) -\
#             beta*S[i]*I[i]/N[i]*dt - S[i]*mort1 + (1-PR[0])*gamma*I[i]*dt +\
#             (R[i]*mort[1] - R[i+1] + gamma*I[i]*PR[0]*dt) +\
#             (V[i]*mort[1] - V[i+1] + nat*N[i]*x*PV[0])

#   N[i+1] = S[i+1] + I[i+1] + R[i+1] + V[i+1]
    N[i+1] = N[i] + N[i]*(nat-mort1)

    print_flag += ntau + nxi

print('%g/%g'%(tmax, tmax))
print()

## Plotting data
print('Plotting data...')

plt.rcParams.update({'font.size': 14})

plt.figure()
plt.plot(t[t0:], S[t0:], 'b', label='Susceptible')
plt.plot(t[t0:], I[t0:], 'r', label='Infectious')
plt.plot(t[t0:], R[t0:], 'g', label='Recovered')
plt.plot(t[t0:], V[t0:], 'k', label='Vaccinated')
plt.title('All compartments')
plt.xlabel('t')
plt.grid()
plt.legend()
plt.tight_layout()
if results_dir is not None:
    plt.savefig(results_dir+"/img_all."+saveformat)

plt.figure()
plt.plot(t[t0:], S[t0:], 'b')
plt.title('Susceptible')
plt.xlabel('t')
plt.grid()
plt.tight_layout()
if results_dir is not None:
    plt.savefig(results_dir+"/img_S."+saveformat)


plt.figure()
plt.plot(t[t0:], I[t0:], 'r')
plt.title('Infectious')
plt.xlabel('t')
plt.grid()
plt.tight_layout()
if results_dir is not None:
    plt.savefig(results_dir+"/img_I."+saveformat)
ind = argrelextrema(I, np.greater)[0]  # indices of peak infectious
plt.scatter(t[ind], I[ind], c='r')
for i in ind:
    plt.annotate('t=%.2g'%t[i], (t[i] + tmax/100, I[i]))
del ind
if results_dir is not None:
    plt.savefig(results_dir+"/img_I_peaks."+saveformat)

plt.figure()
plt.plot(t[t0:], R[t0:], 'g')
plt.title('Recovered')
plt.xlabel('t')
plt.grid()
plt.tight_layout()
if results_dir is not None:
    plt.savefig(results_dir+"/img_R."+saveformat)


plt.figure()
plt.plot(t[t0:], V[t0:], 'k')
plt.title('Vaccinated')
plt.xlabel('t')
plt.grid()
plt.tight_layout()
if results_dir is not None:
    plt.savefig(results_dir+"/img_V."+saveformat)

plt.figure()
plt.plot(t[t0:], N[t0:])
plt.title('Total population')
plt.xlabel('t')
plt.grid()
plt.tight_layout()
if results_dir is not None:
    plt.savefig(results_dir+"/img_N."+saveformat)


plt.figure()
plt.plot(S[t0:], I[t0:])
plt.title('S-I interaction')
plt.xlabel('Susceptible')
plt.ylabel('Infectious')
ind = argrelextrema(I, np.greater)[0][:3]  # 3 indices of peak infectious
plt.scatter(S[ind], I[ind])
for i in ind:
    plt.annotate('t=%.2g'%t[i], (S[i], I[i]))
del ind
plt.grid()
plt.tight_layout()
if results_dir is not None:
    plt.savefig(results_dir+"/img_S-I."+saveformat)


if results_dir is not None:
    np.savetxt(results_dir+"/all_S.txt",  S, fmt='%g')
    np.savetxt(results_dir+"/all_I.txt",  I, fmt='%g')
    np.savetxt(results_dir+"/all_R.txt",  R, fmt='%g')
    np.savetxt(results_dir+"/all_V.txt",  V, fmt='%g')
    np.savetxt(results_dir+"/all_N.txt",  N, fmt='%g')
    np.savetxt(results_dir+"/all_t.txt",  t, fmt='%g')
    np.savetxt(results_dir+"/all_tau.txt",  mort, fmt='%g')
    np.savetxt(results_dir+"/all_xi.txt",  mort, fmt='%g')



"""
plt.figure()
plt.plot(tau, PR)
plt.title('Recovered immunity waning curve')
plt.xlabel('t')
plt.grid()
plt.tight_layout()

plt.figure()
plt.plot(tau, PR)
plt.title('Vaccinated immunity waning curve')
plt.xlabel('t')
plt.grid()
plt.tight_layout()
"""

plt.show()

print()
print('Interesting numbers. There should be outbreak if the first is greater.')
print('%.4f'%(1 - mu*x*(np.sum(PV)*dt)))
print('%.4f'%((gamma + muM)/beta)) # Something like reproduction number

print()

print('Check for last term of P_R,\nshould be less than 1e-6: %.2g'%PR[-1])
print('The same for P_V: %.2g'%PV[-1])
print()

print('Done.')

############
### Dump ###
############
# Enter on #
# own risk #
############

" This is commented, so it won't run "
" Uncomment on own risk "
"""
# Other method to express I(t) analytically
from scipy.integrate import cumtrapz
plt.figure()
plt.plot(t[t0:],
         I0 * np.exp(\
                (beta * np.concatenate(([0], cumtrapz(S[t0:]/N[t0:])*dt)))\
                - (gamma+muM)*t[t0:]),
         label="Variable coeffitient solution, trapz rule"
        )
plt.plot(t[t0:],
         I0 * np.exp(\
                beta * np.cumsum(S[t0:]/N[t0:])*dt\
                - (gamma+muM)*t[t0:]),
         label="Variable coeffitient solution, leftpoint rule"
        )
plt.plot(t[t0:], I[t0:], label="Euler method solution")
plt.title("I(t)")
plt.xlabel("t")
plt.ylabel("I(t)")
plt.tight_layout()
plt.legend()
plt.grid()
plt.yscale("log")
" It's not very effective "
" You cannot mix analytical and discrete solutions. "
" Actually, you can, but do it wisely. "



# Animation of S-I interaction

import matplotlib.animation as animation

# semilogy version
fig = plt.figure()
plt.xlim(np.min(S)*.9, np.max(S)*1.1)
plt.ylim(np.min(I[t0:])/10, np.max(I)*10)
plt.grid()
point = plt.scatter(S[t0], I[t0])
line, = plt.semilogy(S[t0:t0], I[t0:t0])
plt.title("t = %.2f"%t[t0])
plt.xlabel('S(t)')
plt.ylabel('I(t)')
plt.tight_layout()

def anim_update(i):
    line.set_data(S[t0:t0+i], I[t0:t0+i])
    plt.title("t = %.2f"%t[t0+i])
    point.set_offsets((S[t0+i], I[t0+i]))
    #plt.savefig("anim%05d"%i)
    
anim = animation.FuncAnimation(fig, anim_update, range(0, nt-t0, 2),
                               interval=10)
plt.show()

# ordinary version
fig = plt.figure()
plt.xlim(np.min(S)*.9, np.max(S)*1.1)
plt.ylim(-np.max(I)*.1, np.max(I)*1.1)
plt.grid()
line, = plt.plot(S[t0:t0+1], I[t0:t0+1])
point = plt.scatter(S[t0], I[t0])
plt.title("t = %.2f"%t[t0])
plt.xlabel('S(t)')
plt.ylabel('I(t)')
plt.tight_layout()

def anim_update(i):
    line.set_data(S[t0:t0+i+1], I[t0:t0+i+1])
    plt.title("t = %.2f"%t[t0+i])
    point.set_offsets((S[t0+i], I[t0+i]))

anim = animation.FuncAnimation(fig, anim_update, range(0, nt-t0, 2),
                               interval=10)
plt.show()


"""
