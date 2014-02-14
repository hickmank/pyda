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

import inspect
import numpy as np

# Import Data Assimilation class
from pyda.assimilation.assimilation_smoother import DA_smoother

# Import Ensemble Generation class
from pyda.ensemble_generator.SIRensemble import SIRensemble

# Import analysis class
from pyda.analysis_generator.kf.enkf1 import ENKF1

class SIR_DA_SMOOTH(DA_smoother):
    # Defines the array of observations corresponding to a generated ensemble.
    # Observation = (measurement size)x(Ensemble Size) numpy array
    def Model2DataMap(self,Ensemble):
        # In the SIR example we observe just the final Infected
        # proportion at the last ensemble entry.
        Observation = Ensemble[-1]
        return Observation
       
    # Attribute to define the data-error covariance matrix.
    def DataCovInit(self):
        # In the SIR example the data noise is assumed to be a scalar
        # that is constant at each observation. We use an array so
        # that the 'shape' of the data covariance is understood by
        # numpy in a linear algebra sense.
        self.DataCov = np.array([[np.power(self.data_noise,2.0)]])

if __name__ == '__main__':
    # Specify ensemble generation method
    ensemble_method = SIRensemble()

    # Specify analysis method
    analysis_method = ENKF1()

    # Input parameters to specify setup of problem
    data_noise = 0.0025
    data_lag = 2
    Horizon = 200.0
    SimDim = 2
    EnSize = 100
    DataFileName = './data/SIRdata.dat'
    
    # Specify data assimilation method
    DA_method = SIR_DA_SMOOTH(DataFileName,data_noise,data_lag,Horizon,EnSize,SimDim,ensemble_method,analysis_method)

    # Read/Write initialization/parametrization file to correct place.
    DA_method.param_read('./data/SIRsampleparams.dat')
    DA_method.param_write('./param.0.dat')
    
    # Specify assimilation routine parameters
    InitialTime = 0.0
    Ntimestep = 20.0
    Horizon_timesteps = 20.0

    # Run data assimilation routine
    DA_method.DArun(Ntimestep,InitialTime,Horizon_timesteps)
