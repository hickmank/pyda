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
# Module of functions for different visualizations of a data
# assimilation process. 

import numpy as np 
import matplotlib.pyplot as plt 

# NOTE: This visualization assumes that the data being assimilated is
# a scalar.  Uses percentile (100.0*Quantile) cone visualization as
# opposed to trajectory visualization. This will provide a decent
# starting point for designing application specific visualizations for
# data assimilation with ODE systems.
def ode_DA_vis1(SimDim,DataFileName,EnsembleFileName,AnalysisFileName):
    # First load respective files into NumPy arrays and set ensemble size
    DataArray = np.loadtxt(DataFileName,delimiter='\t',skiprows=1)
    Ndata_pts = DataArray.shape[0]

    # Pull out data time
    datatime = DataArray[:,0]

    # Pull out data timeseries
    DataSeries = DataArray[:,1]

    # Define percentiles
    pLev = [5.0, 25.0, 50.0, 75.0, 95.0]

    # Import analysis ensemble data
    AnalysisArray = np.loadtxt(AnalysisFileName)

    # Grab analysis time
    analysistime = AnalysisArray[:,0]

    # Pull array of last simulation dimension of runs 
    # NOTE: This should be fixed to either specify which simulation
    # dimension to graph or cycle through each simulation dimension
    # and create many subplots/plots.
    AnalysisSeries = AnalysisArray[:,SimDim::SimDim]

    # Import ensemble data
    EnsembleArray = np.loadtxt(EnsembleFileName)

    # Grab analysis time
    ensembletime = EnsembleArray[:,0]

    # Pull array of last simulation dimension of runs
    # NOTE: This should be fixed to either specify which simulation
    # dimension to graph or cycle through each simulation dimension
    # and create many subplots/plots.
    EnsembleSeries = EnsembleArray[:,SimDim::SimDim]

    # Calculate ensemble percentile curves
    EnPcnt = np.percentile(EnsembleSeries,pLev,axis=1)

    # Calculate analysis quantile curves
    AnPcnt = np.percentile(AnalysisSeries,pLev,axis=1)

    # Only plot one group of analysis ensemble runs 
    plt.figure(1)

    # Plot 90% cone
    plt.subplot(1,2,1)
    plt.fill_between(ensembletime,EnPcnt[0],EnPcnt[4],
                     color=(158./255.,202./255.,225./255.),alpha=0.5)
    # Plot 50% cone
    plt.subplot(1,2,1)
    plt.fill_between(ensembletime,EnPcnt[1],EnPcnt[3],
                     color=(49./255.,130./255.,189./255.),alpha=0.5)
    # Plot median forecast
    plt.subplot(1,2,1)
    plt.plot(ensembletime,EnPcnt[2],'r-',linewidth=2, label='Median Forecast')
    # Plot data over ensemble plot
    plt.subplot(1,2,1)
    plt.plot(datatime, DataSeries, 'yD', label='Data')
    # Label the plot
    plt.title('ODE Data Assimilation (Ensemble)')
    plt.ylabel('Simulation')
    plt.xlabel('Time')
    plt.xlim([0.0,datatime[-1]])
    plt.legend()

    # Plot 90% cone
    plt.subplot(1,2,2)
    plt.fill_between(analysistime,AnPcnt[0],AnPcnt[4],
                     color=(158./255.,202./255.,225./255.),alpha=0.5)
    # Plot 50% cone
    plt.subplot(1,2,2)
    plt.fill_between(analysistime,AnPcnt[1],AnPcnt[3],
                     color=(49./255.,130./255.,189./255.),alpha=0.5)
    # Plot median forecast
    plt.subplot(1,2,2)
    plt.plot(analysistime,AnPcnt[2],'r-',linewidth=2, label='Median Forecast')
    # Plot data over ensemble plot
    plt.subplot(1,2,2)
    plt.plot(datatime, DataSeries, 'yD', label='Data')
    # Label the plot
    plt.title('ODE Data Assimilation (Analysis)')
    plt.ylabel('Simulation')
    plt.xlabel('Time')
    plt.xlim([0.0,datatime[-1]])
    plt.legend()

    # Display image
    plt.show()
######################################################################
######################################################################

# NOTE: This visualization assumes that the data being assimilated is
# a scalar.  Uses trajectory visualization. This will provide a decent
# starting point for designing application specific visualizations for
# data assimilation with ODE systems.
def ode_DA_vis2(SimDim,DataFileName,EnsembleFileName,AnalysisFileName):
    # First load respective files into NumPy arrays and set ensemble size
    DataArray = np.loadtxt(DataFileName,delimiter='\t',skiprows=1)
    Ndata_pts = DataArray.shape[0]

    # Pull out data time
    datatime = DataArray[:,0]

    # Pull out data timeseries
    DataSeries = DataArray[:,1]

    # Import analysis ensemble data
    AnalysisArray = np.loadtxt(AnalysisFileName)

    # Grab analysis time
    analysistime = AnalysisArray[:,0]

    # Pull array of last simulation dimension of runs 
    # NOTE: This should be fixed to either specify which simulation
    # dimension to graph or cycle through each simulation dimension
    # and create many subplots/plots.
    AnalysisSeries = AnalysisArray[:,SimDim::SimDim]

    # Import ensemble data
    EnsembleArray = np.loadtxt(EnsembleFileName)

    # Grab analysis time
    ensembletime = EnsembleArray[:,0]

    # Pull array of last simulation dimension of runs
    # NOTE: This should be fixed to either specify which simulation
    # dimension to graph or cycle through each simulation dimension
    # and create many subplots/plots.
    EnsembleSeries = EnsembleArray[:,SimDim::SimDim]

    EnSize = EnsembleSeries.shape[1]

    plt.figure(1)
    for i in range(EnSize):
        # Light Blue: color=(36./255.,164./255.,239./255.)
        plt.plot(ensembletime,EnsembleSeries[:,i],color=(161./255.,218./255.,180./255.),linewidth=.2)
        plt.plot(analysistime,AnalysisSeries[:,i],color=(37./255.,52./255.,148./255.),linewidth=.2)
    
    plt.plot(datatime, DataSeries, 'yD', label='Data')
    # Label the plot
    plt.title('ODE Data Assimilation')
    plt.ylabel('Simulation')
    plt.xlabel('Time')
    plt.xlim([0.0,datatime[-1]])

    # Display image
    plt.show()
######################################################################
######################################################################
