# Simple plot generation of enKF data assimilation on the SIR epidemic
# model.

# Import Data Assimilation utilities
import pyda.utilities.AssimilationVis as DA_vis

# Dimension of simulation output
SimDim = 2
# Filename of data assimilated
DataFileName = "./data/SIRdata.dat"
# Filename of Ensemble to plot
EnsembleFileName = "./ensemble.2.dat"
# Filename of Analysis Ensemble to plot
AnalysisFileName = "./analysis.2.dat"

# Call pyda.utilities function
DA_vis.ode_DA_vis1(SimDim,DataFileName,EnsembleFileName,AnalysisFileName)
