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

# Import the EnsembleGeneratorClass
from .ensemble_generator_class import EnsembleGeneratorClass

# Import Epidemic ODE library from pyda utilities
from ..utilities import epiODElib as epi

class SEIRplusEnsemble(EnsembleGeneratorClass):
    def __init__(self):
        # RETURNS:
        #    EnsArray = (2*Ntimesteps)x(EnSize) numpy array. Column is 
        #               (S(t0), E(t0), I(t0), S(t1), E(t1), I(t1), ..., S(tN), E(tN), I(tN))^T
        #    EnsTime = Ntimsteps numpy vector of time
        self.Name = 'Deterministic rk4 SEIRplus'

    def fwd_propagate(self, Param, start_time, stop_time, Ntimestep):
        # Param = (EnSize)x10 Numpy array of ensemble member parameters
        #         <S0>\t<E0>\t<I0>\t<beta_scale>\t<mu_scale>\t<gamma_scale>\t<nu>\t<alpha>\t<c_scale>\t<w_scale>
        # Ntimestep = number of time steps
        EnSize = Param.shape[0]

        # Define time 
        EnsTime = np.linspace(start_time, stop_time, Ntimestep)

        # Define empty array to append ensemble members to
        EnsArray = np.zeros((3*Ntimestep, EnSize))

        # Generate each of the ensemble members
        for i in range(EnSize):
            # Map search parameters to SEIR variables.
            S0 = Param[i,0]
            E0 = Param[i,1]
            I0 = Param[i,2]
            beta = 1.0/(24.0*Param[i,3])
            mu =  1.0/(24.0*Param[i,4])
            gamma = 1.0/(24.0*Param[i,5])
            nu = Param[i,6]
            alpha= Param[i,7] 
            c = (7.0*24.0)*Param[i,8]
            w = (7.0*24.0)*Param[i,9]

            # SEIR
            y0 = np.array([[S0], [E0], [I0]])
		
            # SEIR-plus
            Xsim = epi.SEIRplusode(y0, EnsTime, beta, mu, gamma, nu, alpha, c, w)
            Xsim = Xsim.transpose()

            # Reshape and write to EnsArray.
            EnsArray[:,i] = Xsim.reshape(3*Ntimestep)

        return [EnsArray, EnsTime]

