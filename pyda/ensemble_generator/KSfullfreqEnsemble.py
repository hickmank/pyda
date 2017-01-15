import inspect
import numpy as np

# Import the EnsembleGeneratorClass
from .ensemble_generator_class import EnsembleGeneratorClass

# Import Epidemic ODE library from pyda utilities
from ..utilities import KSsimulator as KS

class KSfullfreqEnsemble(EnsembleGeneratorClass):
    def __init__(self):
        # RETURNS:
        #    EnsArray = (Ntimesteps)x(Ngrid)x(EnSize) numpy array. Each ensemble member 
        #               has form 
        #               (U(t_0), U(t_1), ..., U(t_N))^T
        #    EnsTime = Ntimesteps numpy vector of time
        self.Name = 'Deterministic ETD-RK4 full frequency KS'

    def fwd_propagate(self, Param, start_time, stop_time, Ntimestep):
        # Param = (EnSize)x(Ngrid+1) Numpy array of ensemble member initial condition
        #         and bifurcation parameter. Rows look like,
        #         <Lparam>\t<U0>
        #          Lparam = (float) bifurcation parameter
        #          U0 = (Ngrid) numpy vector of KS solution at start_time
        # Ntimestep = number of time steps
        EnSize = Param.shape[0]
        
        # Define solution spatial grid
        Ngrid = Param[0,1:].shape[0]

        # Define time 
        stepsize = (stop_time - start_time)/(Ntimestep)
        EnsTime = np.arange(start_time, stop_time + stepsize, stepsize)
        EnsTime = EnsTime[:int(Ntimestep+1)]
        
        # Define empty array to append ensemble members to
        EnsArray = np.zeros((int(Ntimestep+1), Ngrid, EnSize))

        # Generate each of the ensemble members
        for i in range(EnSize):
            # Separate ensemble parameters
            Lparam = Param[i,0]
            U0 = Param[i,1:]
		
            # Full frequency KS propagation
            Usoln = KS.fullfreq(U0, EnsTime, Lparam)

            # Reshape and write to EnsArray.
            EnsArray[:,:,i] = Usoln

        return [EnsArray, EnsTime]

