###############################################################################
###############################################################################
#   Copyright 2014 Kyle S. Hickmann and
#                  The Administrators of the Tulane Educational Fund
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
###############################################################################
###############################################################################

###############################################################################
###############################################################################
# This library contains a suite of functions to generate simulations 
# of stochastic and deterministic disease progressions with and without 
# noise due to model error. 
###############################################################################
###############################################################################
import numpy as np 
import math 

###############################################################################
# Simulate a very basic deterministic SIR system with parameters
# beta = transmission rate
# gamma = recovery rate
# Returns numpy array:
#    Xsim = (2)x(Ntimestep) first row S(t), second row I(t).
def SIRode(y0, time, beta, gamma): 
    # y0 = 2x1 numpy array of initial conditions.
    # time = Ntimestep length numpy array
    # beta = scalar
    # gamma = scalar
    
    Xsim = rk4(SIR_D, y0, time, args=(beta,gamma,))
    Xsim = Xsim.transpose()
    return Xsim

# Derivative for basic SIR system.
def SIR_D(y,t,beta,gamma):
    # y[0] = susceptible
    # y[1] = infected
    dy = np.zeros((2,1))
    
    dy[0] = -(beta)*y[0]*y[1]
    dy[1] = (beta)*y[0]*y[1] - gamma*y[1]
    return dy

###############################################################################
# Simulate a modified deterministic SIR system with parameters built to 
# have a power-law scaling for the contact rate as in,
# "Semi-empirical power-law scaling of new infection rate to model
#  epidemic dynamics with inhomogeneous mixing."
#  Mathematical Biosciences 203 (2006) 301-318
# beta = transmission rate
# gamma = recovery rate
# nu = contact rate power-law
# Returns numpy array:
#    Xsim = (2)x(Ntimestep) first row S(t), second row I(t).
def SIRmod1ode(y0, time, beta, gamma, nu): 
    # y0 = 2x1 numpy array of initial conditions.
    # time = Ntimestep length numpy array
    # beta = scalar
    # gamma = scalar
    # nu = scalar
    
    Xsim = rk4(SIRmod1_D, y0, time, args=(beta,gamma,nu,))
    Xsim = Xsim.transpose()
    return Xsim

# Derivative for SIR system.
def SIRmod1_D(y,t,beta,gamma,nu):
    # y[0] = susceptible
    # y[1] = infected
    dy = np.zeros((2,1))
    
    dy[0] = -(beta)*y[1]*math.pow(y[0],nu)
    dy[1] = (beta)*y[1]*math.pow(y[0],nu) - gamma*y[1]
    return dy

###############################################################################
# Simulate a modified deterministic SIR system with parameters built to 
# have a power-law scaling for the contact rate as in,
# "Semi-empirical power-law scaling of new infection rate to model
#  epidemic dynamics with inhomogeneous mixing."
#  Mathematical Biosciences 203 (2006) 301-318
# With this instance we also introduce a time varying transmission rate 
# similar to that used in, "Modeling the spread of influenza among cities"
# by James M. Hyman and Tara LaForce 2003.
# beta = transmission rate
# gamma = recovery rate
# nu = contact rate power-law
# alpha = amplitude of transmission rate variation 
#         beta varies over [1-alpha,1+alpha]
# c = time of peak transmission rate
# w = width of ramp up and ramp down of transmission

# Returns numpy array:
#    Xsim = (2)x(Ntimestep) first row S(t), second row I(t).
def SIRmod2ode(y0, time, beta, gamma, nu, alpha, c, w): 
    # y0 = 2x1 numpy array of initial conditions.
    # time = Ntimestep length numpy array
    # beta = scalar
    # mu = scalar
    # nu = scalar
    
    Xsim = rk4(SIRmod2_D, y0, time, args=(beta,gamma,nu,alpha,c,w,))
    Xsim = Xsim.transpose()
    return Xsim

# Derivative for SIR system.
def SIRmod2_D(y,t,beta,gamma,nu,alpha,c,w):
    # y[0] = susceptible
    # y[1] = infected
    # t = scalar time point
    dy = np.zeros((2,1))
    
    dy[0] = -(trans_rate(t,beta,alpha,c,w))*y[1]*math.pow(y[0],nu)
    dy[1] = (trans_rate(t,beta,alpha,c,w))*y[1]*math.pow(y[0],nu) - gamma*y[1]
    return dy

###############################################################################
# Simulate a basic deterministic SEIR system with parameters
# beta = transmission rate
# mu = incubation rate
# gamma = recovery rate
# Returns numpy array:
#    Xsim = (3)x(Ntimestep) first row S(t), second row I(t),
#           third row E(t).
def SEIRode(y0, time, beta, mu, gamma): 
    # y0 = 3x1 numpy array of initial conditions.
    # time = Ntimestep length numpy array
    # beta = scalar
    # mu = scalar
    # gamma = scalar
    
    Xsim = rk4(SEIR_D, y0, time, args=(beta,mu,gamma,))
    Xsim = Xsim.transpose()
    return Xsim

# Derivative for basic SEIR system.
def SEIR_D(y,t,beta,mu,gamma):
    # y[0] = susceptible
    # y[1] = exposed
    # y[2] = infected/infectious
    dy = np.zeros((3,1))
    
    dy[0] = -(beta)*y[0]*y[2]
    dy[1] = (beta)*y[0]*y[2] - mu*y[1]
    dy[2] = mu*y[1] - gamma*y[2]
    return dy

###############################################################################
# Simulate a deterministic SEIR system with parameters built to 
# have a power-law scaling for the contact rate as in,
# "Semi-empirical power-law scaling of new infection rate to model
#  epidemic dynamics with inhomogeneous mixing."
#  Mathematical Biosciences 203 (2006) 301-318
# With this instance we also introduce a time varying transmission rate 
# similar to that used in, "Modeling the spread of influenza among cities"
# by James M. Hyman and Tara LaForce 2003.

# Returns numpy array:
#    Xsim = (3)x(Ntimestep) first row S(t), second row I(t),
#           third row E(t).
# Variables are (S0,E0,I0,beta,mu,gamma,nu,alpha,c,w)
# S0 = initial susceptible ratio
# E0 = initial exposed ratio
# I0 = initial infected ratio
# R0 = initial recovered ratio, S0+E0+I0+R0=1
# beta = base transmission rate (S0->E0)
# mu = incubation rate (E0->I0)
# gamma = recovery rate (I0->R0)
# nu = contact power-law scaling
# alpha = amplitude of transmission rate variation 
#         beta varies over [1-alpha,1+alpha]
# c = time of peak transmission rate
# w = width of ramp up and ramp down of transmission
def SEIRplusode(y0, time, beta, mu, gamma, nu, alpha, c, w): 
    # y0 = 3x1 numpy array of initial conditions.
    # time = Ntimestep length numpy array
    # beta = scalar
    # mu = scalar
    # gamma = scalar
    
    Xsim = rk4(SEIRplus_D, y0, time, args=(beta,mu,gamma,nu,alpha,c,w,))
    Xsim = Xsim.transpose()
    return Xsim

# Derivative for SEIR-plus system.
def SEIRplus_D(y,t,beta,mu,gamma,nu,alpha,c,w):
    # y[0] = susceptible
    # y[1] = exposed
    # y[2] = infected/infectious
    dy = np.zeros((3,1))
    
    dy[0] = -(trans_rate(t,beta,alpha,c,w))*y[2]*math.pow(y[0],nu)
    dy[1] = (trans_rate(t,beta,alpha,c,w))*y[2]*math.pow(y[0],nu) - mu*y[1]
    dy[2] = mu*y[1] - gamma*y[2]
    return dy

###############################################################################
# Time variable transmission function used in several of the above models.
# This varies over [beta*(1-alpha),beta*(1+alpha)]
def trans_rate(t,beta,alpha,c,w):
    k = 5.0 # parameter controlling transition length
    m = 4.0 # parameter controlling smoothness
    if (math.fabs(2.0*(t-c)/w) < 1):
        rate = (2*math.pow((1-math.pow(math.fabs(2.0*(t-c)/w),k)),m)) - 1.0
    else:
        rate = -1.0

    return (beta*(1 + alpha*rate))

###############################################################################
###############################################################################
# rk4-algorithm used in several of the above simulations.
#
# Integrate an ODE specified by 
# y' = f(y,t,args), y(t0) = y0, t0 < t < T
# Using fourth order Runge-Kutta on a specified time grid.

# f() must return a numpy column array of outputs of the same
# dimension as that of y. It must take a numpy column array of
# variable values y, a single float time t, and any other arguments.

# The output is a numpy array of y values at the times in t of
# dimensions (len(t),len(y0)).
def rk4(func, y0, t, args=()):
    output = np.zeros((len(t),len(y0)))
    output[0,:] = y0[:,0].copy()
    yiter = y0.copy()

    for i in range(len(t)-1):
        h = t[i+1] - t[i]
        k1 = h*func(yiter, t[i], *args)
        k2 = h*func(yiter + .5*k1, t[i] + .5*h, *args)
        k3 = h*func(yiter + .5*k2, t[i] + .5*h, *args)
        k4 = h*func(yiter + k3, t[i] + h, *args)
        yiter = yiter + 0.16666666*(k1 + 2*k2 + 2*k3 + k4)
        output[i+1,:] = yiter[:,0].copy()

    return output

