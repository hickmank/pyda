"""pyda Data Object for Tracking a Data Assimilation Run

All of pyda is driven through manipulation of an hdf5 file. One of the most
problematic components of data assimilation is tracking all the files, time
steps, parameter ensembles, initialization data, background ensembles,
observation sets, analysis ensembles, etc. pyda deals with this problem by
relying on a uniform data structure, written in hdf5, to represent a single
assimilation experiment.

The data structure, referred to as a `darunfile` is a class that initializes an
hdf5 file and contains methods to manipulate each element of the data set
through reference to that elements place in the data assimilation experiment.

"""

"""Goals of the darunfile:

- One file == one DA run

- Data stored as (obs. time, observation) pair even though each observation may
  be high dimension

- Data `name` header stored, i.e. "sat1", "sat2", each name corresponds to one
  `(obs. time, observation)` pair

- State ensemble observations also stored under the name of observation as
  `(obs. time, H(s_1), H(s_2), ..., H(s_ensize))`

- Parameter space and state space separated:

  * Parameter space stored under names "par1", "conductivity", etc. 

  * Parameter ensemble stored as `(obs. time, p_1, p_2, ..., p_ensize)`

  * State space stored under model name "K-S", "L63", "SIR", etc. 

  * State ensemble stored as `(obs. time, s_1, s_2, ..., s_ensize)`

- All time-series used during assimilation should represent the time of observations

- Miscellaneous variables stored, i.e. ensemble size, state dimension, DA
  method, experiment date

- Simulation consists of `(sim. time, s_1, s_2, ..., s_ensize)`

- `obs. time` is a subset of `sim. time`

- Observation error covariance must be stored

- Inflation coefficients must be stored

- Choose not to deal with localization except perhaps through the observation operator

"""

import numpy
import h5py

class darunfile(object):
    """Class to create, manage, and store a data assimilation run.

    This class contains the intialization of a `darunfile`. Methods in this
    class include ways to read and write from each of the data-sets used during
    data assimilation.

    Attributes:
        filename (str): Absolute path of file name as a string with `hdf5` extension
        damethod (str): Name of the assimilation method used, i.e. `enkf`.
        date (str): Date of the experiment `MM-DD-YYYY:HHHH`
        ensize (int): ensemble size
        state_dim (int): dimension of the state variable
        par_dim (int): dimension of parameters adjusted during assimilation

    """
    
    def __init__(self, filename, damethod, date, ensize, state_dim, par_dim=0):
        """Initialize darun attributes

        Args:
            filename (str): Absolute path of file name as a string with `hdf5` extension
            damethod (str): Name of the assimilation method used, i.e. `enkf`.
            date (str): Date of the experiment `MM-DD-YYYY:HHHH`
            ensize (int): ensemble size
            state_dim (int): dimension of the state variable
            par_dim (int): dimension of parameters adjusted during assimilation

        """
        self.filename = filename
        self.damethod = damethod
        self.date = date
        self.ensize = ensize
        self.state_dim = state_dim
        self.par_dim = par_dim

    def create_file(self)
        self.darunfile = h5py.File(self.filename, "w")
