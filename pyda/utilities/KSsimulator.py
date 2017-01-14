# Function definitions to simulate 1D Kuramoto-Sivashinsky solutions 
# using Exponential Time-differencing with Fourth order
# Runge-Kutta. These functions can then be used in a data 
# assimilation routine.

# u_t = -u*u_x - u_xx - u_xxxx, periodic BCs on [-PI*L, PI*L]
# computation is based on v = fft(u), so linear term is diagonal
# compare p27.m in Trefethen, "Spectral Methods in MATLAB", SIAM 2000
# AK Kassam and LN Trefethen, July 2002

# To simplify the Fourier transform we use 
# u*u_x = 0.5*(d/dx)(u^2)

# Fourier form of equation with 
#      u(x,t) = \sum_k u_k(t) exp(ikx), k = n/L, 
#      du_k/dt = (k^2)*(1 - k^2) u_k + (1j*k/2) \sum_i u_i(t) u_{k-i}(t)
#      du_n/dt = [(n/L)^2]*(1 - (n/L)^2) u_n + (1j*n/2*L) \sum_{j \in Z} u_j(t) u_{n-j}(t)

# Then first 0 <= n < L Fourier modes are unstable at u_n(t) = 0

import numpy as np
import numpy.fft as npFT

# Returns numpy array:
#    Usoln = (Ntimestep)x(N) Each row is the solution at a specified time
# 
#    The solution is given on the discretized interval
#       x = -PI*Lparam + 2*PI*Lparam*np.arange(1.0, float(N+1))/float(N)
# Input:
#    U0 = (N) numpy array representing initial condition 
#         on interval [-PI*Lparam, PI*Lparam]
#    time = (Ntimesteps) numpy array of the times solution is computed for
#    Lparam = Bifurcation parameter
def fullfreq(U0, time, Lparam):
    PI = np.pi
    
    # Spatial grid
    # N should be an integer power of 2.0
    N = U0.shape[0]

    # Temporal grid
    Ntimesteps = time.shape[0]

    # Specify FULL number of Fourier modes to simulate from.
    # 1.0 <= FmodeMax <= float(N/2)
    # Number of modes will be FmodeMax+1 since we always 
    # include the zero frequency.
    FmodeMax = float(N/2.0) 

    # Precompute various ETDRK4 scalar quantities:
    # Time step
    h = time[1] - time[0]

    # Wave numbers 
    # Full, N/2+1, wave numbers up to Nyquist frequency are 
    # 0.0 <= k <= N/(2*Lparam)
    k = np.arange(0., FmodeMax + 1.0)/Lparam

    # Fourier multipliers
    L = np.power(k, 2.0) - np.power(k, 4.0)
    E = np.exp(h*L)
    E2 = np.exp(h*L/2.0)

    # Define the terms in the contour integrals for ETDRK4
    # Number points on upper half of unit circle 
    M = 32
    # Define sample of path r = np.exp(1j*a), 0.0 <= a < PI
    r = np.exp(1j*PI*(np.arange(0.0, float(M))/float(M)))
    # Parameterize path in resolvent (t + Lh), t = path around unit circle
    LR = h*np.tile(L, (M, 1)).transpose() + np.tile(r, (FmodeMax+1, 1));

    # Contour integral approximations for ETD
    Q = h*(np.mean((np.exp(LR/2.0) - 1.0)/LR, 1))
    f1 = h*(np.mean((-4.0 - LR + np.exp(LR)*(4.0 - 3.0*LR + np.power(LR,2.0)))/np.power(LR, 3.0), 1))
    f2 = h*(np.mean((2.0 + LR + np.exp(LR)*(-2.0 + LR))/np.power(LR, 3.0), 1))
    f3 = h*(np.mean((-4.0 - 3.0*LR - np.power(LR, 2.0) + np.exp(LR)*(4.0 - LR))/np.power(LR, 3.0), 1))

    Q = Q.real
    f1 = f1.real
    f2 = f2.real
    f3 = f3.real

    # Main time-stepping loop:
    # Initialize full frequency solution
    Usoln = U0
    v = npFT.rfft(U0) # This introduces numerical error in IC
    g = -0.5*1j*k

    # Main ETD_RK4 time stepping
    for n in range(Ntimesteps-1):
        Nv = g*npFT.rfft(np.power(npFT.irfft(v), 2.0))

        a = E2*v + Q*Nv
        Na = g*npFT.rfft(np.power(npFT.irfft(a), 2.0))

        b = E2*v + Q*Na
        Nb = g*npFT.rfft(np.power(npFT.irfft(b), 2.0))

        c = E2*a + Q*(2.0*Nb - Nv)
        Nc = g*npFT.rfft(np.power(npFT.irfft(c), 2.0))

        v = E*v + f1*Nv + 2.0*f2*(Na + Nb) + f3*Nc

        # Update full solution
        u = npFT.irfft(v, N) # Ensure u has correct size
        Usoln = np.vstack([Usoln, u])

    return Usoln

# Returns numpy array:
#    Usoln = (Ntimestep)x(N) Each row is the solution at a specified time
# 
#    The solution is given on the discretized interval
#       x = -PI*Lparam + 2*PI*Lparam*np.arange(1.0, float(N+1))/float(N)
# Input:
#    U0 = (N) numpy array representing initial condition 
#         on interval [-PI*Lparam, PI*Lparam]
#    time = (Ntimesteps) numpy array of the times solution is computed for
#    Lparam = Bifurcation parameter
#    FmodeMax = (float) Specify number of Fourier modes to simulate from.
#               1.0 <= FmodeMax <= float(N/2)
#               Number of modes will be FmodeMax+1 since we always 
#               include the zero frequency.
def lowfreq(U0, time, Lparam, FmodeMax):
    PI = np.pi
    
    # Spatial grid
    # N should be an integer power of 2.0
    N = U0.shape[0]

    # Temporal grid
    Ntimesteps = time.shape[0]

    # Precompute various ETDRK4 scalar quantities:
    # Time step
    h = time[1] - time[0]

    # Wave numbers 
    # Full, N/2+1, wave numbers up to Nyquist frequency are 
    # 0.0 <= k <= N/(2*Lparam)
    k = np.arange(0., FmodeMax + 1.0)/Lparam

    # Fourier multipliers
    L = np.power(k, 2.0) - np.power(k, 4.0)
    E = np.exp(h*L)
    E2 = np.exp(h*L/2.0)

    # Define the terms in the contour integrals for ETDRK4
    # Number points on upper half of unit circle 
    M = 32
    # Define sample of path r = np.exp(1j*a), 0.0 <= a < PI
    r = np.exp(1j*PI*(np.arange(0.0, float(M))/float(M)))
    # Parameterize path in resolvent (t + Lh), t = path around unit circle
    LR = h*np.tile(L, (M, 1)).transpose() + np.tile(r, (FmodeMax+1, 1));

    # Contour integral approximations for ETD
    Q = h*(np.mean((np.exp(LR/2.0) - 1.0)/LR, 1))
    f1 = h*(np.mean((-4.0 - LR + np.exp(LR)*(4.0 - 3.0*LR + np.power(LR,2.0)))/np.power(LR, 3.0), 1))
    f2 = h*(np.mean((2.0 + LR + np.exp(LR)*(-2.0 + LR))/np.power(LR, 3.0), 1))
    f3 = h*(np.mean((-4.0 - 3.0*LR - np.power(LR, 2.0) + np.exp(LR)*(4.0 - LR))/np.power(LR, 3.0), 1))

    Q = Q.real
    f1 = f1.real
    f2 = f2.real
    f3 = f3.real

    # Main time-stepping loop:
    # Initialize full frequency solution
    Usoln = U0
    v = npFT.rfft(U0) # This introduces numerical error in IC
    v = v[0:(FmodeMax+1)]
    g = -0.5*1j*k

    # Main ETD_RK4 time stepping
    for n in range(Ntimesteps-1):
        Nv = g*npFT.rfft(np.power(npFT.irfft(v), 2.0))

        a = E2*v + Q*Nv
        Na = g*npFT.rfft(np.power(npFT.irfft(a), 2.0))

        b = E2*v + Q*Na
        Nb = g*npFT.rfft(np.power(npFT.irfft(b), 2.0))

        c = E2*a + Q*(2.0*Nb - Nv)
        Nc = g*npFT.rfft(np.power(npFT.irfft(c), 2.0))

        v = E*v + f1*Nv + 2.0*f2*(Na + Nb) + f3*Nc

        # Update full solution
        u = npFT.irfft(v, N) # Ensure u has correct size
        Usoln = np.vstack([Usoln, u])

    return Usoln
