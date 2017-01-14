import inspect
import numpy as np

class AnalysisGeneratorClass(object):
    # This serves as a template for the user of pyda to implement
    # their own analysis generation. The initial conditions and
    # parameters used in the ensemble are contained passed to the
    # analysis scheme in one array. Hence the initialization
    # conditions are treated as parameters. 

    # When the analysis is used it must MINIMALLY be passed
    # information about: 
    #                  - ensemble array
    #                  - ensemble observation
    #                  - data array
    #                  - data covariance
    #                  - parameter/initialization array

    # INPUT: {numpy arrays}
    #      <Data Array> = (measurement size)x(ensemble size)
    #      <Data Covariance> = (measurement size)x(measurement size)
    #      <Ensemble Array> = (simulation size)x(ensemble size)
    #      <Parameter Array> = (parameter size)x(ensemble size)
    #      <Ensemble Observation Array> = (measurement size)x(ensemble size)
    #
    # RETURN: {numpy arrays}
    #      <Analysis Array> = (simulation size)x(ensemble size)
    #      <Analysis Parameter Array> = (parameter size)x(ensemble size)
    def __init__(self):
        self.Name = 'Analysis Scheme Name'

    # Returns the analysis ensemble array and the analysis parameter array.
    # AnsArray = (Ntimestep*SimulationDimension)x(EnSize) numpy array
    # AnsParamArray = (Parameter Size + Initialization Size)x(EnSize) numpy array
    def create_analysis(self,Data,DataCov,ParamArray,EnsArray,ObsArray):
        # For an example implementation look at
        # enkf*.py.
        self.ParamArray = ParamArray
        self.EnSize = ParamArray.shape[1]
        self.Data = Data
        self.DataCov = DataCov
        self.ObsArray = ObsArray
        self.EnsArray
        
        # return [AnsArray,AnsParamArray]
        raise NotImplementedError(inspect.stack()[0][3])
