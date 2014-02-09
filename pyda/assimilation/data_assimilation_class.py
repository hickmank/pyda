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

class DataAssimilationClass(object):
    # This class is the main machinery of the pyda data assimilation
    # package. It controls how the parameter/initialization array is
    # formed and how the data is read. This class also controls how
    # the data covariance is defined and defines the model-to-data
    # map. The forecast horizon is defined in this class and the
    # interaction of the analysis-filter and ensemble generation. It
    # also controls reading and writing of the parameter, ensemble,
    # and analysis arrays.

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

    # Initializes the parameter/initialization array.
    # ParamArray = (param size + initialization size)x(Ensemble Size) numpy array
    def param_initialization(self):
        # This can either be a routine to generate an initial
        # parameter/initialization array or it can read a file of
        # pre-generated parameter/initialization sets. It must then
        # create the self.Param = Param numpy array that
        # will be passed to the ensemble generation and analysis
        # generation classes. 
        
        # self.Param = 
        raise NotImplementedError(inspect.stack()[0][3])

    # Attribute to write a parameter array to a file. 
    def param_write(self,ParamFileName):
        # np.savetxt(ParamFileName,self.Param,fmt='%5.5f',delimiter='\t')
        raise NotImplementedError(inspect.stack()[0][3])

    # Attribute to read a parameter array to a file. 
    def param_read(self,ParamFileName):
        # self.Param = np.loadtxt(ParamFileName,delimiter='\t',skiprows=1)
        raise NotImplementedError(inspect.stack()[0][3])

    # Defines the array of observations corresponding to a generated ensemble.
    # Observation = (measurement size)x(Ensemble Size) numpy array
    def Model2DataMap(self,Ensemble):
        # Defines the mapping of the simulation to the data. 
        raise NotImplementedError(inspect.stack()[0][3])

    # Attribute to write an ensemble array to a file. This may involve
    # reshaping the ensemble array depending on the model and how the
    # file is to be used later. The ensemble time vector is always the
    # left most column.
    def ensemble_write(self,Ensemble,EnsTime,EnsFileName):
        # EnsembleWrite is combined numpy array.
        # np.savetxt(EnsFileName,EnsembleWrite,fmt='%5.5f',delimiter=' ')
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
    def DArun(self,Ntimestep,InitialTime):
        # Data Assimilation process consists of 3 steps: 
        # 1.) Ensemble generation 
        # 2.) Analysis generation 
        # 3.) Write analysis/ensemble arrays and parameter array 
        # The process is then repeated until the data is exhausted. At this 
        # point a forecast is made until the final Horizon.

        # This will need to be tweeked slightly depending on the exact
        # way the data is meant to be assimilated at each step and the
        # nature of the analysis step used.
        
        # Initialize start time
        start_time = InitialTime

        # Initialize Parametrization/Initialization
        self.param_initialization()

        # Initialize Data Covariance
        self.DataCovInit()

        for i in range(self.Data.shape[0]):
            # 1.) Ensemble Generation
            stop_time = self.DataTime[i]
            Ensemble = self._ensemble.fwd_propagate(self.Param,start_time,stop_time,Ntimestep)        
            Observation = self.Model2DataMap(Ensemble)
            
            # Write ensemble array
            # EnsFileName = 
            self.ensemble_write(Ensemble,self._ensemble.EnsTime,EnsFileName)            

            # Increment start time
            start_time = stop_time
            
            # Define data array used in ensemble analysis
            # DataArray = (observation size)x(ensemble size) numpy array created by Data[i,:]
            #        and DataCov
            # 2.) Analysis generation
            [Analysis,AnalysisParam] = self._analysis.create_analysis(DataArray,self.DataCov,self.Param,Ensemble,Observation)
            self.Param = AnalysisParam
            # AnsFileName = 
            # ParamFileName = 
            self.param_write(ParamFileName)
            self.ensemble_write(Analysis,self._ensemble.EnsTime,AnsFileName)            

        # Forecast to end horizon
        # Horizon_timesteps = 
        Forecast = self._ensemble.fwd_propagate(self.Param,start_time,self.Horizon,Horizon_timesteps)
        # Write forecast array
        # ForecastFileName = 
        self.ensemble_write(Forecast,self._ensemble.EnsTime,ForecastFileName)            

