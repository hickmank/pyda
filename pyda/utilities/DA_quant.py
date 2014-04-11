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

# Module of functions to evaluate the effectiveness of Data
# Assimilation methods. Mainly the evaluation of effectiveness is done
# by computing the KL-divergence between the analysis ensemble and the
# background ensemble along with the data likelihood under the
# analysis ensemble. This can be done explicitly for the Kalman Filter
# schemes which assume Gaussianity. For the sequential Monte Carlo and
# Particle Filter methods we use Kernel Density Approximation of the
# distributions.

import numpy as np 
import math

# First we compute the Kullback-Leibler Divergence for two Gaussian
# distributions. We will need to pass the ensemble and analysis
# measurement distributions to do this. The mean and covariance are
# then formed from these. An alternative would be to pass the mean and
# covariance matrices.
#      <Ensemble Observation Arrays> = (measurement size)x(ensemble size)
def ensemble_KLdiv(ensemble_observations, analysis_observations):
    # Collect data sizes
    EnSize = ensemble_observations.shape[1]
    MeaSize = ensemble_observations.shape[0]

    # Calculate analysis and ensemble means
    Emean = (1./float(EnSize))*ensemble_observations.sum(1)
    Amean = (1./float(EnSize))*analysis_observations.sum(1)

    # Compute covariance matrices
    dE = ensemble_observations - np.tile(Emean.reshape(MeaSize,1),(1,EnSize))
    dA = analysis_observations - np.tile(Amean.reshape(MeaSize,1),(1,EnSize))

    Ecov = (1./float(EnSize-1))*np.dot(dE,dE.transpose())
    Acov = (1./float(EnSize-1))*np.dot(dA,dA.transpose())

    # Now compute D_{KL}(analysis | ensemble) for the Gaussians.
    # We compute this in three parts
    KL1 = np.trace(np.linalg.solve(Ecov,Acov))
    KL2 = np.dot((Amean - Emean),np.linalg.solve(Ecov,(Amean - Emean)))
    KL3 = np.linalg.det(Acov)/np.linalg.det(Ecov)
    KLdiv = 0.5*(KL1 + KL2 - math.log(KL3) - MeaSize)

    return KLdiv

# We return the likelihood of a data point given an analysis ensemble
# of observations. The analysis observation ensemble is used to
# compute an estimated mean and covariance. A Gaussian density with
# this mean and covariance is then evaluated at the data point to
# return the likelihood.
def GuassLikelihood(data, analysis_observations):
    # Collect data sizes
    EnSize = analysis_observations.shape[1]
    MeaSize = analysis_observations.shape[0]

    # Calculate analysis mean
    Amean = (1./float(EnSize))*analysis_observations.sum(1)

    # Compute covariance matrix
    dA = analysis_observations - np.tile(Amean.reshape(MeaSize,1),(1,EnSize))
    Acov = (1./float(EnSize-1))*np.dot(dA,dA.transpose())

    # Compute Gaussian likelihood
    Coef = math.sqrt(math.pow((2*math.pi),MeaSize)*np.linalg.det(Acov))

    Arg = 0.5*(np.dot((data - Amean),np.linalg.solve(Acov,(data - Amean))))

    GaussLikelihood = math.exp(-Arg)/Coef

    return GaussLikelihood
