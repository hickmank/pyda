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
# Module to visualize one step of a data assimilation process. Uses
# quantile cone visualization as opposed to trajectory visualization. 

# NOTE: This vsualization assumes that the data being assimilated is a scalar.

# This will provide a decent starting point for designing application
# specific visualizations for data assimilation with ODE systems.

# Use:
# AssimilationVis(SimDim,DataFileName,EnsembleFileName,AnalysisFileName)

import numpy as np 
import numpy.random as rn
import math 
import matplotlib.pyplot as plt 

def AssimilationVis(SimDim,DataFileName,EnsembleFileName,AnalysisFileName):
    # First load respective files into NumPy arrays and set ensemble size
    DataArray = np.loadtxt(DataFileName,delimiter='\t')
    Ndata_pts = DataArray.shape[0]

    # Pull out data time
    datatime = DataArray[:,0]

    # Pull out data timeseries
    DataSeries = DataArray[:,1]

    # Define quantiles
    QntLev = [0.05, 0.25, 0.5, 0.75, 0.95]

    # Import analysis ensemble data
    AnalysisArray = np.loadtxt(AnalysisFileName)

    # Grab analysis time
    analysistime = AnalysisArray[:,0]

    # Pull array of last simulation dimension of runs
    AnalysisSeries = AnalysisArray[:,SimDim::SimDim]

    # Import ensemble data
    EnsembleArray = np.loadtxt(EnsembleFileName)

    # Grab analysis time
    ensembletime = EnsembleArray[:,0]

    # Pull array of last simulation dimension of runs
    EnsembleSeries = EnsembleArray[:,SimDim::SimDim]

    # Calculate ensemble quantile curves
    EnQnt = mst.mquantiles(EnsembleSeries,prob=QntLev,axis=1)

    # Calculate analysis quantile curves
    AnQnt = mst.mquantiles(AnalysisSeries,prob=QntLev,axis=1)

    # Only plot one group of analysis ensemble runs 
    plt.figure(1)

    # Plot 90% cone
    plt.subplot(1,2,1)
    plt.fill_between(ensembletime,EnQnt[:,0],AnQnt[:,4],
                     color=(158./255.,202./255.,225./255.),alpha=0.5)
    # Plot 50% cone
    plt.subplot(1,2,1)
    plt.fill_between(ensembletime,EnQnt[:,1],AnQnt[:,3],
                     color=(49./255.,130./255.,189./255.),alpha=0.5)
    # Plot median forecast
    plt.subplot(1,2,1)
    plt.plot(ensembletime,EnQnt[:,2],'r-',linewidth=2, label='Median Forecast')
    # Plot data over ensemble plot
    plt.subplot(1,2,1)
    plt.plot(datatime, DataSeries, 'yD', label='Data')

    # Plot 90% cone
    plt.subplot(1,2,2)
    plt.fill_between(analysistime,AnQnt[:,0],AnQnt[:,4],
                     color=(158./255.,202./255.,225./255.),alpha=0.5)
    # Plot 50% cone
    plt.subplot(1,2,2)
    plt.fill_between(analysistime,AnQnt[:,1],AnQnt[:,3],
                     color=(49./255.,130./255.,189./255.),alpha=0.5)
    # Plot median forecast
    plt.subplot(1,2,2)
    plt.plot(analysistime,AnQnt[:,2],'r-',linewidth=2, label='Median Forecast')
    # Plot data over ensemble plot
    plt.subplot(1,2,2)
    plt.plot(datatime, DataSeries, 'yD', label='Data')

    # Label the plot
    plt.title('U.S. Forecast, 2013-2014 ILI up to Week 3 (no Wiki-data)')
    plt.ylabel('% Visits for ILI')
    plt.xlabel('Epidemic Week')
    plt.xlim([analysistime[0],analysistime[-1]])
    plt.ylim([0,10]) # If you want to fix y-axis
    plt.xticks(Time[(14-1)::14], map(int,epiWk))
    plt.legend()
    plt.show()

