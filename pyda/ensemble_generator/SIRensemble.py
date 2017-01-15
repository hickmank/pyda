import inspect
import numpy as np

# Import the EnsembleGeneratorClass
from .ensemble_generator_class import EnsembleGeneratorClass

# Import Epidemic ODE library from pyda utilities
from ..utilities import epiODElib as epi

class SIRensemble(EnsembleGeneratorClass):
    def __init__(self):
        # RETURNS:
        #    EnsArray = (2*Ntimesteps)x(EnSize) numpy array. Column is 
        #               (S(t0), I(t0), S(t1), I(t1), ..., S(tN), I(tN))^T
        #    EnsTime = Ntimsteps numpy vector of time
        self.Name = 'Deterministic rk4 SIR'

    def fwd_propagate(self,Param,start_time,stop_time,Ntimestep):
        # Initialization includes the parameter array, start time, stop time, and the number of timesteps
        # Param = (EnSize)x(4) Numpy array of ensemble member parameters
        #              <S0>\t<I0>\t<beta>\t<gamma>
        EnSize = Param.shape[0]
        EnsTime = np.linspace(start_time,stop_time,Ntimestep)

        # Define empty array to append ensemble members to
        EnsArray = np.zeros((2*Ntimestep,EnSize))

        # Generate each of the ensemble members
        for i in range(EnSize):
            # Read file ICs to variable names
            S0 = Param[i,0]
            I0 = Param[i,1]
            beta = Param[i,2]
            gamma = Param[i,3]

            # Initial Conditions
            y0 = np.array([[S0],[I0]])
 
            # Simulate SIR
            Xsim = epi.SIRode(y0, EnsTime, beta, gamma)
            Xsim = Xsim.transpose()

            # Reshape and write to EnsArray.
            EnsArray[:,i] = Xsim.reshape(2*Ntimestep)

        return [EnsArray, EnsTime]

class SIRensembleILI(EnsembleGeneratorClass):
    def __init__(self):
        # RETURNS:
        #    EnsArray = (2*Ntimesteps)x(EnSize) numpy array. Column is 
        #               (S(t0), I(t0), S(t1), I(t1), ..., S(tN), I(tN))^T
        #    EnsTime = Ntimsteps numpy vector of time
        self.Name = 'Deterministic rk4 SIR for ILI'

    def fwd_propagate(self,Param,start_time,stop_time,Ntimestep):
        # Initialization includes the parameter array, start time, stop time, and the number of timesteps
        # Param = (EnSize)x(4) Numpy array of ensemble member parameters
        #              <S0>\t<I0>\t<beta>\t<gamma>\t<c>
        EnSize = Param.shape[0]
        EnsTime = np.linspace(start_time, stop_time, Ntimestep)

        # Define empty array to append ensemble members to
        EnsArray = np.zeros((2*Ntimestep, EnSize))

        # Generate each of the ensemble members
        for i in range(EnSize):
            # Read file ICs to variable names
            S0 = Param[i,0]
            I0 = Param[i,1]
            beta = 1.0/(24.0*Param[i,2])
            gamma = 1.0/(24.0*Param[i,3])

            # Initial Conditions
            y0 = np.array([[S0],[I0]])
 
            # Simulate SIR
            Xsim = epi.SIRode(y0, EnsTime, beta, gamma)
            Xsim = Xsim.transpose()

            # Reshape and write to EnsArray.
            EnsArray[:,i] = Xsim.reshape(2*Ntimestep)

        return [EnsArray, EnsTime]


