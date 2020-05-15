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

save = False
saveformat = "png"

## Time parameters settings
print('Setting model parameters...')
print('Setting time parameters')
dt     =  .01      # time step
tmax   =  100       # time horizon for t (in years)
taumax =  100       # time horizon for tau (in years)
ximax  =  taumax    # time horizon for xi (in years)
# taumax, ximax should be large enough that PR, PV < 1e-8

# population and disease parameter setting
print('Setting population and disease parameters')
beta = 50      # transmission rate
x = 0.8        # vaccintaion rate
#x = .5324877  # marginal value of x for beta = 50, nu = .02, gamma = 36.5
nu = .01       # birth rate
mu = nu       # mortality rate
d = 10/365     # avg_years_of_infection
gamma = 1/d

# Under-the-hood variables
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


# pre-calculated prob. of not dying for t=0*dt, 1*dt, 2*dt, ...
# used for computing R and V
# mort = (1 - mu*dt)**(np.arange(nt))
# nat = nu*dt
mort  = np.exp(-mu * np.arange(nt)*dt)  # continuous equivalent of mu*dt
nat   = 1 - np.exp(-nu * dt)
mort1 = 1 - mort[1]

rec   = 1 - np.exp(-gamma * dt)  # continuous equivalent od gamma*dt

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

# For testing with SIRS model
#theta = 0.03602081608379648
#theta = 0.01
theta = 1/(np.sum(PV) * dt)
PV = np.exp(-tau * theta)
PR = np.exp(-xi * theta)

# Convolution P function, refined for deaths
PRmort = PR[::-1] * mort[-(nt-ntau+1)::-1]
PVmort = PV[::-1] * mort[-(nt-nxi +1)::-1]


print('time step: %.2g'%dt)
print('tmax: %.2g'%tmax)
print('taumax: %.2g'%taumax)
print('ximax: %.2g'%ximax)

print('beta: %.2g'%beta)
print('x: %.1g'%x)
print('nu: %.2g'%nu)
print('mu: %.2g'%mu)
print('gamma: %.2g'%gamma)
print('GMT0: %.3g'%GMT0)
print('w_tau: %.2g'%w_tau)
print('w_xi: %.2g'%w_xi)
print('sigma: %.2g'%sigma)
print('Ccrit: %.2g'%Ccrit)
print('R0:\n%.4f\n'%((1 - I0/N0 - nu*x*np.inner(PV, mort[:nxi])*dt)*beta/(gamma+mu)))


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
            f.write('nu: %.2g\n'%nu)
            f.write('mu: %.2g\n'%mu)
            f.write('gamma: %.2g\n'%gamma)
            f.write('GMT0: %.3g\n'%GMT0)
            f.write('w_tau: %.2g\n'%w_tau)
            f.write('w_xi: %.2g\n'%w_xi)
            f.write('sigma: %.2g\n'%sigma)
            f.write('Ccrit: %.2g\n\n'%Ccrit)

            f.write('min(PR): %.2g\n\n'%PR[-1])
            f.write('min(PV): %.2g\n\n'%PV[-1])

            f.write('1 - nu*x*(np.sum(PV)*dt):\n')
            f.write('%.4f\n'%(1 - nu*x*(np.sum(PV)*dt)))

            f.write('(gamma + mu)/beta:\n')
            f.write('%.4f\n'%((gamma + mu)/beta))

            f.write('R0:\n%.4f\n'%((1 - I0/N0 - nu*x*np.sum(PVmort)*dt)*\
                                     beta/(gamma+mu)))

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
V[:t0+1] = N0 * nat * x *(np.inner(PV[:], mort[:nxi]) )

# For now, zero boundary for R is used:
# long time ago there were no recovered.
R[:t0+1] = 0

# Susceptible is everybody who is not vaccinated, recovered nor infected
S[:t0+1] = N[:t0+1] - V[:t0+1] - R[:t0+1] - I[:t0+1]

print()
print('Interesting numbers. There should be outbreak if the first is greater.')
print('%.4f'%(1 - nu*x*(np.sum(PV)*dt)))
print('%.4f'%((gamma + mu)/beta))

print('Being happy for python not indexing from 1 like matlab...')

print("Saving initial condition...")
if results_dir is not None:
    np.savetxt(results_dir+"/par_PR.txt",  PR, fmt='%g')
    np.savetxt(results_dir+"/par_PV.txt",  PV, fmt='%g')
    np.savetxt(results_dir+"/par_PRmort.txt",  PRmort, fmt='%g')
    np.savetxt(results_dir+"/par_PVmort.txt",  PVmort, fmt='%g')
    np.savetxt(results_dir+"/par_mort.txt",  mort, fmt='%g')

print_flag = 0  # just for printing progress (print every few sec)

print("Exact effective reproduction number: %.4f"%(S[t0] * beta / (N[t0] * (gamma + mu))))

## Time loop
print()
print('Starting time loop...')
print()
for i in range(t0, nt-1):
    if print_flag > 2**31:
        print('%.3g/%.3g'%(t[i], tmax))
        print_flag = 0
    I[i+1] = I[i] + I[i] * beta*S[i]/N[i]*dt - I[i] * gamma * dt - I[i] *mort1

    R[i+1] = gamma * dt * np.inner(PRmort, I[(i-ntau+1):(i+1)])

    V[i+1] = nat*x* np.inner(PVmort, N[(i-nxi+1):(i+1)])

    #S[i+1] = S[i] + nat*N[i] - S[i]*mort1 -\
    #         beta*S[i]*I[i]/N[i]*dt + gamma*I[i]*dt +\
    #         (R[i]*mort[1] - R[i+1]) +\
    #         (V[i]*mort[1] - V[i+1])
    S[i+1] = (N[i] + N[i]*(nat-mort1)) - I[i+1] - R[i+1] - V[i+1]

#    S[i+1] = S[i] + nat*N[i]*(1-x) + nat*N[i]*x*(1-PV[0]) -\
#             beta*S[i]*I[i]/N[i]*dt - S[i]*mort1 + (1-PR[0])*gamma*I[i]*dt +\
#             (R[i]*mort[1] - R[i+1] + gamma*I[i]*PR[0]*dt) +\
#             (V[i]*mort[1] - V[i+1] + nat*N[i]*x*PV[0])

#   N[i+1] = S[i+1] + I[i+1] + R[i+1] + V[i+1]
    N[i+1] = N[i] + N[i]*(nat-mort1)

    print_flag += ntau + nxi

print('%g/%g'%(tmax, tmax))
print()


##############################################################################

## Plotting data
print('Plotting data...')


plt.rcParams.update({'font.size': 14,
                     'font.family': 'serif',
                     'figure.figsize': (11.52, 6.48)
                     })

plt.figure()
plt.plot(t[t0:], S[t0:], 'b', label='Susceptible')
plt.plot(t[t0:], I[t0:], 'r', label='Infectious')
plt.plot(t[t0:], R[t0:], 'g', label='Recovered')
plt.plot(t[t0:], V[t0:], 'k', label='Vaccinated')
plt.title("Our model")
#plt.title('All compartments')
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
plt.plot(t[t0:], ((gamma + mu)/beta)*N[t0:], "k:")
if results_dir is not None:
    plt.savefig(results_dir+"/img_S_cirt."+saveformat)

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
plt.plot(t[t0:], N[t0:], ":", label="Explicit", color="C0")
plt.plot(t[t0:], S[t0:] + I[t0:] + R[t0:] + V[t0:], label="Sum", color="C0")
plt.title('Total population')
plt.xlabel('t')
plt.grid()
plt.tight_layout()
plt.legend()
if results_dir is not None:
    plt.savefig(results_dir+"/img_N."+saveformat)


# TODO: Fix peaks plotting
# what if there are less than 3 peaks
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

plt.figure()
plt.plot(tau, PR)
plt.title('Recovered immunity waning curve')
plt.xlabel("$\\tau$")
plt.ylabel("$P_R(\\tau)$")
plt.grid()
plt.tight_layout()
if results_dir is not None:
    plt.savefig(results_dir+"/img_PR."+saveformat)

plt.figure()
plt.plot(tau, PV)
plt.title('Vaccinated immunity waning curve')
plt.xlabel("$\\tau$")
plt.ylabel("$P_V(\\tau)$")
plt.grid()
plt.tight_layout()
if results_dir is not None:
    plt.savefig(results_dir+"/img_PV."+saveformat)

plt.figure()
plt.plot(t[t0:], np.ones_like(t[t0:]), ":k")
plt.plot(t[t0:], beta/(gamma+mu) * S[t0:]/N[t0:])
plt.title("Effective reproduction number")
plt.xlabel("t")
plt.ylabel("$\\mathcal{R}(t)$")
plt.grid()
plt.tight_layout()
if results_dir is not None:
    plt.savefig(results_dir+"/img_Repr."+saveformat)


if results_dir is not None:
    np.savetxt(results_dir+"/all_S.txt",  S, fmt='%g')
    np.savetxt(results_dir+"/all_I.txt",  I, fmt='%g')
    np.savetxt(results_dir+"/all_R.txt",  R, fmt='%g')
    np.savetxt(results_dir+"/all_V.txt",  V, fmt='%g')
    np.savetxt(results_dir+"/all_N.txt",  N, fmt='%g')
    np.savetxt(results_dir+"/all_t.txt",  t, fmt='%g')
    np.savetxt(results_dir+"/all_tau.txt",  mort, fmt='%g')
    np.savetxt(results_dir+"/all_xi.txt",  mort, fmt='%g')


plt.show()

print()
print('Interesting numbers. There should be outbreak if the first is greater.')
print('%.4f'%(1 - nu*x*(np.sum(PVmort)*dt)))
print('%.4f'%((gamma + mu)/beta))

print()

print('Check for last term of P_R,\nshould be << 1:\n %.2g'%PR[-1])
print('The same for P_V:\n %.2g'%PV[-1])
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
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

try:
    theta = theta
except NameError:
    #theta = 0.03602081608379648
    theta = 1 / (np.sum(PV) * dt)

def SIRSmodel(t, SIR, x=x, mu=mu, nu=nu, beta=beta, gamma=gamma, theta=theta):
    S, I, R = SIR
    N = S+I+R
    dS = theta*R - beta*S*I/N + nu*N*(1-x) - mu*S
    dI = beta*S*I/N - gamma*I - mu*I
    dR = gamma*I - mu*R - theta*R + nu*N*x
    return np.array((dS, dI, dR))

#SIRS = odeint(fun=SIRS, y0=(S[t0], I[t0], R[t0] + V[t0]),
#              t_span=t[t0:], tfirst=True)
"""

"""
SIRS = solve_ivp(SIRSmodel, [0, tmax], (S[t0], I[t0], R[t0] + V[t0]),
                      t_eval=t[t0:], method="RK45")



plt.figure()
plt.plot(t[t0:], SIRS.y[0,:], color="blue", label="Suseptible")
plt.plot(t[t0:], SIRS.y[1,:], color="red", label="Infectious")
plt.plot(t[t0:], SIRS.y[2,:], color="green", label="Recovered")
plt.title("SIRS model solution")
plt.xlabel("t")
plt.grid()
plt.legend()
plt.tight_layout()


plt.figure()
plt.plot(t[t0:], SIRS.y[0,:], label="SIRS")
plt.plot(t[t0:], S[t0:], label="Our model")
plt.title("Susceptible")
plt.xlabel("t")
plt.ylabel("S(t)")
plt.grid()
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(t[t0:], SIRS.y[1,:], label="SIRS")
plt.plot(t[t0:], I[t0:], label="Our model")
plt.title("Infectious")
plt.xlabel("t")
plt.ylabel("I(t)")
plt.grid()
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(t[t0:], SIRS.y[2,:], label="SIRS")
plt.plot(t[t0:], R[t0:] + V[t0:], label="Our model")
plt.title("Recovered")
plt.xlabel("t")
plt.ylabel("R(t)")
plt.grid()
plt.legend()
plt.tight_layout()


def SIRVSmodel(t, SIRV, x=x, mu=mu, nu=nu, beta=beta, gamma=gamma, theta=theta):
    S, I, R, V = SIRV
    N = S+I+R+V
    dS = theta*(R + V) + nu*N*(1-x) - beta*S*I/N - mu*S
    dI = beta*S*I/N - gamma*I - mu*I
    dR = gamma*I - mu*R - theta*R
    dV = nu*N*x - mu*V - theta*V
    return np.array((dS, dI, dR, dV))

init_cond = np.array([N[t0], 0, 0, 0])
for i in range(50):
    init_cond = solve_ivp(SIRVSmodel, [0, tmax], init_cond,
                      t_eval=t[t0:], method="RK45").y[:, -1]
init_cond[0] -= 1; init_cond[1] += 1

init_cond = np.array([78429.66032383,     1.        ,     0.        , 21569.33967617])

SIRVS = solve_ivp(SIRVSmodel, [0, tmax], init_cond,
                      t_eval=t[t0:], method="RK45")


SIRVS = solve_ivp(SIRVSmodel, [0, tmax], (S[t0], I[t0], R[t0], V[t0]),
                      t_eval=t[t0:], method="RK45")


plt.figure()
plt.plot(t[t0:], SIRVS.y[0,:], color="blue", label="Suseptible")
plt.plot(t[t0:], SIRVS.y[1,:], color="red", label="Infectious")
plt.plot(t[t0:], SIRVS.y[2,:], color="green", label="Recovered")
plt.plot(t[t0:], SIRVS.y[3,:], color="black", label="Vaccinated")
plt.title("SIRVS model solution")
plt.xlabel("t")
plt.grid()
plt.legend()
plt.tight_layout()


plt.figure()
plt.plot(t[t0:], SIRVS.y[0,:], label="SIRVS")
plt.plot(t[t0:], S[t0:], label="Our model")
plt.title("Susceptible")
plt.xlabel("t")
plt.ylabel("S(t)")
plt.grid()
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(t[t0:], SIRVS.y[1,:], label="SIRVS")
plt.plot(t[t0:], I[t0:], label="Our model")
plt.title("Infectious")
plt.xlabel("t")
plt.ylabel("I(t)")
plt.grid()
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(t[t0:], SIRVS.y[2,:], label="SIRVS")
plt.plot(t[t0:], R[t0:], label="Our model")
plt.title("Recovered")
plt.xlabel("t")
plt.ylabel("R(t)")
plt.grid()
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(t[t0:], SIRVS.y[3,:], label="SIRVS")
plt.plot(t[t0:], V[t0:], label="Our model")
plt.title("Vaccinated")
plt.xlabel("t")
plt.ylabel("V(t)")
plt.grid()
plt.legend()
plt.tight_layout()
"""

"""
# Runge-Kutta solutions without solver

SIRS_RK = np.zeros((3, len(t) - t0))
SIRS_RK[:, 0] = (S[t0], I[t0], R[t0] + V[t0])

print("Calculating Runge-Kutta solution...")
for i in range(len(t)-t0-1):
    k1 = dt * SIRSmodel(t[i+t0],        SIRS_RK[:, i])
    k2 = dt * SIRSmodel(t[i+t0] + dt/2, SIRS_RK[:, i] + k1/2)
    k3 = dt * SIRSmodel(t[i+t0] + dt/2, SIRS_RK[:, i] + k2/2)
    k4 = dt * SIRSmodel(t[i+t0] + dt,   SIRS_RK[:, i] + k3)

    SIRS_RK[:, i+1] = SIRS_RK[:, i] + k1/6 + k2/3 + k3/3 + k4/6

print("Plotting...")
plt.figure()
plt.plot(t[t0:], SIRS_RK[0,:], color="blue", label="Susceptible")
plt.plot(t[t0:], SIRS_RK[1,:], color="red", label="Infectious")
plt.plot(t[t0:], SIRS_RK[2,:], color="green", label="Recovered")
plt.title("SIRS solution")
plt.xlabel("t")
plt.grid()
plt.legend()
plt.tight_layout()
if results_dir is not None:
    plt.savefig(results_dir+"/img_SIRS_all."+saveformat)

plt.figure()
plt.plot(t[t0:], SIRS_RK[0,:], color="blue", label="SIRS")
plt.title("Susceptible")
plt.xlabel("t")
plt.ylabel("S(t)")
plt.grid()
plt.tight_layout()
if results_dir is not None:
    plt.savefig(results_dir+"/img_SIRS_S."+saveformat)
plt.plot(t[t0:], S[t0:], color="black", linestyle=":", label="Our model")
plt.legend()
if results_dir is not None:
    plt.savefig(results_dir+"/img_SIRS_S_comp."+saveformat)

plt.figure()
plt.plot(t[t0:], SIRS_RK[1,:], color="red", label="SIRS")
plt.title("Infectious")
plt.xlabel("t")
plt.ylabel("I(t)")
plt.grid()
plt.tight_layout()
if results_dir is not None:
    plt.savefig(results_dir+"/img_SIRS_I."+saveformat)
plt.plot(t[t0:], I[t0:], color="black", linestyle=":", label="Our model")
plt.legend()
if results_dir is not None:
    plt.savefig(results_dir+"/img_SIRS_I_comp."+saveformat)

plt.figure()
plt.plot(t[t0:], SIRS_RK[2,:], color="green", label="SIRS")
plt.title("Recovered")
plt.xlabel("t")
plt.ylabel("R(t)")
plt.grid()
plt.tight_layout()
if results_dir is not None:
    plt.savefig(results_dir+"/img_SIRS_R."+saveformat)
plt.plot(t[t0:], R[t0:] + V[t0:],
         color="black", linestyle=":", label="Our model")
plt.legend()
if results_dir is not None:
    plt.savefig(results_dir+"/img_SIRS_R_comp."+saveformat)

plt.figure()
plt.plot(t[t0:], N0 * np.exp(t[t0:] * (nu-mu)), ":", label="Explicit", color="C0")
plt.plot(t[t0:], np.sum(SIRS_RK, axis=0), label="SIRS", color="C0")
plt.title('Total population')
plt.xlabel('t')
plt.grid()
plt.tight_layout()
if results_dir is not None:
    plt.savefig(results_dir+"/img_SIRS_N."+saveformat)
plt.plot(t[t0:], S[t0:] + I[t0:] + R[t0:] + V[t0:], color="C1", label="Our model")
plt.legend()
if results_dir is not None:
    plt.savefig(results_dir+"/img_SIRS_N_comp."+saveformat)

print("Done.")

"""
##############################################################################
"""
SIRVS_RK = np.zeros((4, len(t) - t0))
SIRVS_RK[:, 0] = (S[t0], I[t0], R[t0], V[t0])

for i in range(len(t)-t0-1):
    k1 = dt * SIRVSmodel(t[i+t0],        SIRVS_RK[:, i])
    k2 = dt * SIRVSmodel(t[i+t0] + dt/2, SIRVS_RK[:, i] + k1/2)
    k3 = dt * SIRVSmodel(t[i+t0] + dt/2, SIRVS_RK[:, i] + k2/2)
    k4 = dt * SIRVSmodel(t[i+t0] + dt,   SIRVS_RK[:, i] + k3)

    SIRVS_RK[:, i+1] = SIRVS_RK[:, i] + k1/6 + k2/3 + k3/3 + k4/6

plt.figure()
plt.plot(t[t0:], SIRVS_RK[0,:], label="SIRVS")
plt.plot(t[t0:], S[t0:], "--", label="Our model")
plt.title("Susceptible")
plt.xlabel("t")
plt.ylabel("S(t)")
plt.grid()
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(t[t0:], SIRVS_RK[1,:], label="SIRVS")
plt.plot(t[t0:], I[t0:], "--", label="Our model")
plt.title("Infectious")
plt.xlabel("t")
plt.ylabel("I(t)")
plt.grid()
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(t[t0:], SIRVS_RK[2,:], label="SIRVS")
plt.plot(t[t0:], R[t0:], "--", label="Our model")
plt.title("Recovered")
plt.xlabel("t")
plt.ylabel("R(t)")
plt.grid()
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(t[t0:], SIRVS_RK[3,:], label="SIRVS")
plt.plot(t[t0:], V[t0:], "--", label="Our model")
plt.title("Vaccinated")
plt.xlabel("t")
plt.ylabel("V(t)")
plt.grid()
plt.legend()
plt.tight_layout()


# Other method to express I(t) analytically
from scipy.integrate import cumtrapz
plt.figure()
plt.plot(t[t0:],
         I0 * np.exp(\
                (beta * np.concatenate(([0], cumtrapz(S[t0:]/N[t0:])*dt)))\
                - (gamma+mu)*t[t0:]),
         label="Variable coeffitient solution, trapz rule"
        )
plt.plot(t[t0:],
         I0 * np.exp(\
                beta * np.cumsum(S[t0:]/N[t0:])*dt\
                - (gamma+mu)*t[t0:]),
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
##############################################################################
"""
ab_S[:t0+1] = 1e5
ab_I[:t0+1] = 0
ab_R[:t0+1] = 0
ab_V[:t0+1] = 0

ab_S[:t0+1] = ab_S[-1]
ab_I[:t0+1] = ab_I[-1]
ab_R[:t0+1] = ab_R[-1]
ab_V[:t0+1] = ab_V[-1]


### Adams-Bashforth

from scipy import integrate

# Copy initial condition
ab_S = S.copy(); ab_S[t0+1:] = 0
ab_I = I.copy(); ab_I[t0+1:] = 0
ab_R = R.copy(); ab_R[t0+1:] = 0
ab_V = V.copy(); ab_V[t0+1:] = 0
ab_N = N0 * np.exp(t * (nu-mu))

dI = np.zeros_like(ab_I)




# Calculate I[t0+2] with two steps of size h
for i in range(t0, t0+2):
    dI[i] = beta*ab_S[i]*ab_I[i]/ab_N[i] - ab_I[i] * (gamma + mu)

    ab_I[i+1] = ab_I[i] +  dt * dI[i]

    ab_R[i+1] = rec * np.inner(PRmort, ab_I[(i-ntau+1):(i+1)])

    ab_V[i+1] = nat*x* np.inner(PVmort, ab_N[(i-nxi+1):(i+1)])

    ab_S[i+1] = ab_N[i+1] - ab_I[i+1] - ab_R[i+1] - ab_V[i+1]

# We use Richardson's extrapolation now for I[t0+2]
ab_I[t0+2] = 2 * ab_I[t0+2] - (ab_I[t0] + 2*dt * dI[t0])


print_flag = 0
for i in range(t0+2, nt-1, 2):
#    if print_flag > 2**31:
#        print('%.3g/%.3g'%(t[i], tmax))
#        print_flag = 0

#   predictor

    dI[i] = beta*ab_S[i]*ab_I[i]/ab_N[i] - ab_I[i] * (gamma + mu)

    ab_I[i+2] = ab_I[i] + dt * (3*dI[i] - 1*dI[i-2])

    z = PRmort[::2] * ab_I[(i-ntau+1):(i+1):2]  # Temporal variable. We split this at t=0 in order to get better results
    if i - ntau + 1 < t0:
        ab_R[i+2] = gamma * integrate.simps(z[-(1 + (i-t0)//2):]) * 2 * dt + \
                    gamma * integrate.simps(z[:-(1 + (i-t0)//2)]) * 2 * dt
    else:
        ab_R[i+2] = gamma * integrate.simps(z) * 2 * dt

    ab_V[i+2] = mu * x * integrate.simps(PVmort[::2] * ab_N[(i-nxi+1):(i+1):2]) * 2 * dt
    """
    ab_R[i+2] = gamma * integrate.simps(PRmort[::2] * ab_I[(i-ntau+1):(i+1):2]) * 2 * dt

    ab_V[i+2] = mu * x * integrate.simps(PVmort[:i-index:2] * ab_N[(index):(i+1):2]) * 2 * dt
    """

    print_flag += ntau + nxi

    ab_S[i+2] = ab_N[i+2] - ab_I[i+2] - ab_R[i+2] - ab_V[i+2]

#   Corrector

    for j in range(10):  # We will apply correction 10 times.
        if i%100 == 0:
            print(ab_I[i+2])

        dI[i+2] = beta*ab_S[i+2]*ab_I[i+2]/ab_N[i+2] - ab_I[i+2] * (gamma + mu)

        #ab_I[i+2] = ab_I[i] + dt * (dI[i] + dI[i+2])   # 2-step orrector

        ab_I[i+2] = ab_I[i] + dt * (5*dI[i+2] + 8*dI[i] - dI[i-2]) / 6  # 3-step corrector

        z = PRmort[::2] * ab_I[(i-ntau+1):(i+1):2]  # Temporal variable. We split this at t=0 in order to get better results
        if i - ntau + 1 < t0:
            ab_R[i+2] = gamma * integrate.simps(z[-(1 + (i-t0)//2):]) * 2 * dt + \
                        gamma * integrate.simps(z[:-(1 + (i-t0)//2)]) * 2 * dt
        else:
            ab_R[i+2] = gamma * integrate.simps(z) * 2 * dt

        ab_V[i+2] = mu * x * integrate.simps(PVmort[::2] * ab_N[(i-nxi+1):(i+1):2]) * 2 * dt
        """
        ab_R[i+2] = gamma * integrate.simps(PRmort[::2] * ab_I[(i-ntau+1):(i+1):2]) * 2 * dt

        ab_V[i+2] = mu * x * integrate.simps(PVmort[:i-index:2] * ab_N[(index):(i+1):2]) * 2 * dt
        """

        ab_S[i+2] = ab_N[i+2] - ab_I[i+2] - ab_R[i+2] - ab_V[i+2]

        print_flag += ntau + nxi
    if i%100 == 0:
        print()


plt.figure()
plt.plot(t[t0::2], ab_S[t0::2], 'b', label='Susceptible')
plt.plot(t[t0::2], ab_I[t0::2], 'r', label='Infectious')
plt.plot(t[t0::2], ab_R[t0::2], 'g', label='Recovered')
plt.plot(t[t0::2], ab_V[t0::2], 'k', label='Vaccinated')
plt.xlabel("t")
plt.grid()
plt.legend()
plt.tight_layout()
"""