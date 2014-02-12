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
import numpy.random as rn

# Import Data Assimilation class
from .data_assimilation_class import DataAssimilationClass

class DA_current2horizon(DataAssimilationClass):
    # This implements a data assimilation scheme in which only the
    # most recent data point is used to adjust the ensemble and
    # parametrization. The ensemble is then propagated from the most
    # recent data point to the horizon at each assimilation
    # step. Also, the data array passed to the analysis step is
    # perturbed by samples from normal distribution controlled by the
    # data covariance matrix. For deterministic analysis schemes and
    # some particle filters this will not be necessary.

    # NOTE: This just takes care of how data is used in the
    # assimilation. For a specific data assimilation problem the
    # model2data map and data covariance matrix must be specified.

    # The class is initialized by reading in the data to arrays.
    # Data = (Ndata_pts)x(measurement size) numpy array
    # DataTime = (Ndata_pts) numpy vector
    # data_noise = either scalar standard deviation of data error 
    #              or vector of standard deviations in data error 
    #              at each data point
    # EnSize = Integer specifying size of ensemble
    # SimDim = Dimension of the simulation at each timestep
    # Horizon = Time to propagate ensemble forecast until after last data 
    #           point is assimilated
    # EnsembleClass = object of type EnsembleGeneratorClass(object) that will
    #                 be called to generate the ensemble
    # AnalysisClass = object of type AnalysisGeneratorClass(object) that will 
    #                 be called to form the analysis ensemble
    def __init__(self,DataFileName,data_noise,Horizon,EnSize,SimDim,EnsembleClass,AnalysisClass):
        tmpData = np.loadtxt(DataFileName,delimiter='\t')
        self.DataTime = tmpData[:,0]
        self.Data = tmpData[:,1:]
        self.data_noise = data_noise
        self.Horizon = Horizon
        self._ensemble = EnsembleClass
        self._analysis = AnalysisClass
        
        # Important sizes and dimensions
        self.EnSize = EnSize
        self.SimDim = SimDim
        self.ObsDim = self.Data.shape[1]

    # Attribute to read a parameter array to a file. There must be
    # SimDim first columns that contain values of the Ensemble State
    # variables to propagate model from.
    def param_read(self,ParamFileName):
        self.Param = np.loadtxt(ParamFileName,delimiter='\t')

    # Attribute to write a parameter array to a file. 
    def param_write(self,ParamFileName):
        np.savetxt(ParamFileName,self.Param,fmt='%5.5f',delimiter='\t')

    # Attribute to write an ensemble array to a file. The ensemble
    # array is reshaped. The ensemble time vector is the left most
    # column.
    def ensemble_write(self,Ensemble,EnsTime,EnsFileName):
        # EnsembleWrite is combined numpy array.
        Ntimestep = EnsTime.shape[0]
        EnsembleWrite = np.zeros((Ntimestep, self.SimDim*self.EnSize+1))
        EnsembleWrite[:,0] = EnsTime

        for i in range(self.EnSize):
            EnsembleWrite[:,(self.SimDim*i+1):(self.SimDim*(i+1)+1)] = Ensemble[:,i].reshape(Ntimestep,self.SimDim) 

        np.savetxt(EnsFileName,EnsembleWrite,fmt='%5.5f',delimiter=' ')

    # Defines the array of observations corresponding to a generated ensemble.
    # Observation = (measurement size)x(Ensemble Size) numpy array
    def Model2DataMap(self,Ensemble):
        # Defines the mapping of the simulation to the data. 
        raise NotImplementedError(inspect.stack()[0][3])

    # Attribute to define the data-error covariance matrix.
    def DataCovInit(self):
        # The error covariance in the data must be defined. If the
        # data being assimilated at each step is just a scalar then
        # this will just be a scalar. However, if the data being
        # assimilated at each step is a vector the covariance will be
        # a square matrix of equal size.

        # This should use self.data_noise
        # self.DataCov = 
        raise NotImplementedError(inspect.stack()[0][3])

    # Attribute that controls how the data, ensemble generation,
    # analysis, and reading/writing interact.
    # Ntimestep = number of timesteps per data point in ensemble generation
    # InitialTime = time simulation starts before first data point
    def DArun(self,Ntimestep,InitialTime):
        # Data Assimilation process consists of 3 steps: 
        # 1.) Ensemble generation 
        # 2.) Analysis generation 
        # 3.) Write analysis/ensemble arrays and parameter array 
        # The process is then repeated until the data is exhausted. At this 
        # point a forecast is made until the final Horizon.
        
        # Initialize start time
        start_time = InitialTime

        # Initialize Parametrization/Initialization
        ParamFileName = "".join(('./param.0.dat'))
        self.param_read(ParamFileName)

        # Initialize Data Covariance
        self.DataCovInit()

        for i in range(self.Data.shape[0]):
            # 1.) Ensemble Generation to next data point
            stop_time = self.DataTime[i]
            [Ensemble,EnsTime] = self._ensemble.fwd_propagate(self.Param,start_time,stop_time,Ntimestep)      
            Observation = self.Model2DataMap(Ensemble)

            # Generate ensemble forecast to horizon
            Ndatapt_remaining = self.Data[i:,:].shape[0] + 1.0
            [Forecast,CastTime] = self._ensemble.fwd_propagate(self.Param,start_time,self.Horizon,Ntimestep*Ndatapt_remaining)

            # Write ensemble array
            EnsFileName = "".join(('./ensemble.',str(i),'.dat'))
            self.ensemble_write(Forecast,CastTime,EnsFileName)            

            # Increment start time
            start_time = stop_time
            
            # Define data array used in ensemble analysis
            # DataArray = (observation size)x(ensemble size) numpy array created by Data[i,:]
            #        and DataArray            
            # Handle scalar Data Covariance and Matrix cases differently
            if (self.DataCov.ndim == 0):
                DataPerturbation = np.sqrt(self.DataCov)*rn.randn(1,self.EnSize)
            elif (self.DataCov.ndim == 2):
                # Compute SVD of Data Covariance to generate noise
                U, s, V = np.linalg.svd(self.DataCov, full_matrices=False)
                DataPerturbation = np.dot(np.dot(U,np.diag(s)),rn.randn(self.Data.shape[1],self.EnSize))
            else:
                print 'Data Covariance should be matrix or scalar', '\n'

            DataArray = np.tile(self.Data[i,:],(self.EnSize,1)).transpose() + DataPerturbation

            # 2.) Analysis generation
            [Analysis,AnalysisParam] = self._analysis.create_analysis(DataArray,self.DataCov,self.Param,Ensemble,Observation)
            # Update the current state of the model to generate the next ensemble from.
            AnalysisParam[:,:self.SimDim] = Analysis[-self.SimDim:,:].transpose()
            self.Param = AnalysisParam

            # Generate analysis forecast to horizon starting at new data point
            Ndatapt_remaining = self.Data[(i+1):,:].shape[0] + 1.0
            [Forecast,CastTime] = self._ensemble.fwd_propagate(self.Param,start_time,self.Horizon,Ntimestep*Ndatapt_remaining)

            # Write analysis to files
            AnsFileName = "".join(('./analysis.',str(i),'.dat'))
            ParamFileName = "".join(('./param.',str(i+1),'.dat'))
            self.param_write(ParamFileName)
            self.ensemble_write(Forecast,CastTime,AnsFileName)            

