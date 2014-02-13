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

# Import AnalysisGeneratorClass
from ..analysis_generator_class import AnalysisGeneratorClass

# Unlike the ensemble Kalman Filter analysis methods the Particle
# filter and Sequential Monte Carlo methods generate an analysis using
# a weighting method and a resampling method. These are programmed as
# separate attribute functions for the particle filter classes and
# then called when the analysis ensemble is created.

class PF_NAIVE(AnalysisGeneratorClass):
    # This is an implementation of the most basic interpretation of a
    # particle filter
    def __init__(self):
        # INPUT: {numpy arrays}
        #      <Data Array> = (measurement size)x(ensemble size)
        #      <Data Covariance> = (measurement size)x(measurement size)
        #      <Ensemble Array> = (simulation size)x(ensemble size)
        #      <Parameter Array> = (ensemble size)x(parameter size)
        #      <Ensemble Observation Array> = (measurement size)x(ensemble size)
        #
        # RETURN: {numpy arrays}
        #      <Analysis Array> = (simulation size)x(ensemble size)
        #      <Analysis Parameter Array> = (ensemble size)x(parameter size)
        self.Name = 'Naive Particle Filter'

    # In this, most basic, form of the PF the Param array, and
    # Ensemble array are not used to generate weights. Particle
    # weights are only determined by the Data, DatCov, and the EnsObs
    # associatted with each particle. The data likelihood is assumed
    # to be Gaussian.
    def weight(self,Data,DataCov,Ensemble,Observation):
        # Collect data sizes.
        EnSize = Ensemble.shape[1]

        # First create weight array.
        # W is (1)x(EnSize)
        W = np.zeros(EnSize)

        # Calculate data perturbations from ensemble measurements
        # Dpert = (MeasSize)x(EnSize)
        Dpert = Data - Observation

        # Compute inv(DataCov)*Dpert
        # Should be (MeasSize)x(EnSize)
        B = np.linalg.solve(DataCov,Dpert)

        # Calculate un-normalized weight for each particle using observations
        NormArg = np.diag(np.dot(Dpert.transpose(),B))
        W = np.exp(-(0.5)*(NormArg))

        # Now normalize weights
        W = W/np.sum(W)

        # Weight the ensemble
        self.W = W

    # Resampling functions use the Ensemble and Parameter arrays,
    # along with weights calulated with one of the Particle filters,
    # to generate an analysis Ensemble with equal weights.  In this,
    # most basic form of resampling, ensemble members are resampled
    # according to their weights directly. This will cause problematic
    # duplications in parameter samples.
    def resample(self,Ensemble,Param):
        # Get ensemble size
        EnSize = Ensemble.shape[1]

        # Generate resampled indices
        index = range(EnSize)
        resamp_index = np.random.choice(index,size=EnSize,replace=True,p=self.W)

        # Create analysis ensembles
        AnalysisEnsemble = Ensemble[:,resamp_index] 
        AnalysisParams = Param[resamp_index,:]

        return [AnalysisEnsemble,AnalysisParams]

    # Returns the analysis ensemble array and the analysis parameter array.
    # Analysis = (Ntimestep*SimulationDimension)x(EnSize) numpy array
    # AnalysisParam = (Parameter Size + Initialization Size)x(EnSize) numpy array
    def create_analysis(self,Data,DataCov,Param,Ensemble,Observation):
        # Weight the ensemble by data likelihood
        self.weight(Data,DataCov,Ensemble,Observation)
        
        # Resample ensemble by weights
        [Analysis,AnalysisParam] = self.resample(Ensemble,Param)
                
        return [Analysis,AnalysisParam]

