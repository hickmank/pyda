import inspect
import numpy as np
import math

# Import AnalysisGeneratorClass
from ..analysis_generator_class import AnalysisGeneratorClass

class ENKF1_inflation(AnalysisGeneratorClass):
    # This is an implementation of a stochastic ensemble Kalman filter
    # with covariance inflation. Each ensemble member is updated
    # separately and the full ensemble covariance is never formed.
    def __init__(self,rho=1.0):
        # INPUT: {numpy arrays}
        #      <Data Array> = (measurement size)x(ensemble size)
        #      <Data Covariance> = (measurement size)x(measurement size)
        #      <Ensemble Array> = (simulation size)x(ensemble size)
        #      <Parameter Array> = (parameter size)x(ensemble size)
        #      <Ensemble Observation Array> = (measurement size)x(ensemble size)
        #      <rho> = (scalar) covariance inflation parameter. Usually rho >= 1.
        # RETURN: {numpy arrays}
        #      <Analysis Array> = (simulation size)x(ensemble size)
        #      <Analysis Parameter Array> = (ensemble size)x(parameter size)
        self.Name = 'Stochastic Ensemble Kalman Filter with Covariance Inflation'
        self.rho = rho

    # Returns the analysis ensemble array and the analysis parameter array.
    # Analysis = (Ntimestep*SimulationDimension)x(EnSize) numpy array
    # AnalysisParam = (Parameter Size + Initialization Size)x(EnSize) numpy array
    def create_analysis(self,Data,DataCov,Param,Ensemble,Observation):
        # Collect data sizes.
        EnSize = Ensemble.shape[1]
        SimSize = Ensemble.shape[0] 
        MeaSize = Data.shape[0]

        # First combine the Ensemble and Param arrays.
        # A is (SimSize+ParSize)x(EnSize)
        A = np.vstack([Ensemble, Param.transpose()])

        # Calculate ensemble mean
        Amean = (1./float(EnSize))*np.tile(A.sum(1), (EnSize,1)).transpose()

        # Calculate ensemble observation mean
        MeasAvg = (1./float(EnSize))*np.tile(Observation.reshape(MeaSize,EnSize).sum(1), (EnSize,1)).transpose()
        
        # Inflate only the simulation ensemble, not the parameter ensemble 
        A[:(SimSize+1),:] = math.sqrt(self.rho)*(A[:(SimSize+1),:] - Amean[:(SimSize+1),:]) + Amean[:(SimSize+1),:]

        # Inflate the ensemble observations
        Observation = math.sqrt(self.rho)*(Observation - MeasAvg) + MeasAvg

        # Calculate ensemble perturbation from mean
        # Apert should be (SimSize+ParSize)x(EnSize)
        dA = A - Amean

        # Data perturbation from ensemble measurements
        # dD should be (MeasSize)x(EnSize)
        dD = Data - Observation

        # Ensemble measurement perturbation from ensemble measurement mean.
        # S is (MeasSize)x(EnSize)
        S = Observation - MeasAvg

        # Set up measurement covariance matrix
        # COV is (MeasSize)x(MeasSize)
        COV = (1./float(EnSize-1))*np.dot(S,S.transpose()) + DataCov
    
        # Compute inv(COV)*dD
        # Should be (MeasSize)x(EnSize)
        B = np.linalg.solve(COV,dD)

        # Adjust ensemble perturbations
        # Should be (SimSize+ParSize)x(MeasSize)
        dAS = (1./float(EnSize-1))*np.dot(dA,S.transpose())

        # Compute analysis
        # Analysis is (SimSize+ParSize)x(EnSize)
        Analysis = A + np.dot(dAS,B)

        # Separate and return Analyzed Ensemble and Analyzed Parameters.
        AnalysisParam = Analysis[SimSize:,:].transpose()
        Analysis = Analysis[0:SimSize,:]
                
        return [Analysis,AnalysisParam]

