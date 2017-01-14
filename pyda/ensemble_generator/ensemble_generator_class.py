import inspect
import numpy as np

class EnsembleGeneratorClass(object):
    # This serves as a template for the user of pyda to implement
    # their own ensemble generation. It usually calls some simulation
    # routine provided by the user. The initial conditions and
    # parameters used in the ensemble generation are contained in one
    # array. Hence the initialization conditions are treated as
    # parameters.

    def __init__(self):
        # Initialization just defines the object and this will be used
        # by the data assimilation function in the
        # DataAssimilationClass.
        self.Name = 'Ensemble Name'

    # Returns the ensemble array propagated forward from the
    # initialization and parametrization in Param. 
    # EnsArray = (Ntimestep*SimulationDimension)x(EnSize) numpy array    
    def fwd_propagate(self,Param,start_time,stop_time,Ntimestep):
        # For an example implementation look at
        # SIRensemble.py.

        # Initialization includes the parameter array, start time, stop time, and the number of timesteps
        # Param = (param size + initialization size)x(Ensemble Size) numpy array
        # start_time, stop_time = float
        # Ntimestep = integer
        self.Param = Param
        self.start_time = start_time
        self.stop_time = stop_time
        self.Ntimestep = Ntimestep
        self.EnSize = Param.shape[1]
        self.EnsTime = np.linspace(start_time,stop_time,Ntimestep)

        # Usually a function is to be called a number of times equal
        # to self.EnSize corresponding to the different
        # initializations/parametrizations in self.ParamArray. The
        # simulation is then run from self.start_time to
        # self.stop_time for a number of timesteps equal to
        # self.Ntimestep. Regardless of the shape of the array
        # returned by the simulation each simulation output is turned
        # into a column vector of succesive time points and inserted
        # into a column of EnsArray. This is then attached to the
        # EnsembleGeneratorClass object using self.EnsArray =
        # EnsArray.
        raise NotImplementedError(inspect.stack()[0][3])
