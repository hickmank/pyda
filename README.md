pyda: Object Oriented Data Assimilation Framework for Python
============================================================

pyda separates ensemble generation and filtering/analysis into
separate class objects. These are then used together in an
assimilation class object. This allows the user to code the ensemble
generation, assimilation, and filtering/analysis steps
separately. Allowing concentration on the element of the assimilation
process they are interested in refining. This also allows a user to
code up their particular ensemble generation method and use the
assimilation and filtering classes we have provided to perform the
assimilation. If you have written a time-dependent simulation in
python then pyda will handle the data assimilation problem. There is
minimal support for visualization and assimilation evaluation as well.

This package includes:
----------------------
     - Classes to generate an ensemble of runs from a simulation
     - Classes to perform various flavors of Ensemble Kalman filtering
     - Classes to perform various particle filtering and sequential 
       monte carlo filtering 
     - Classes to control interaction between ensemble generation, data, 
       and analysis for data assimilation
     - Multiple examples to experiment with data assimilation
     - Functions to visualize data assimilation process occurring in the exmples
     - Functions to evaluate effectiveness of data assimilation process

Dependencies:
-------------
	 - Numpy
	 - Matplotlib
	 - Scipy
	 - h5py

Quick Start:
------------

Run

```
python setup.py install
```
      
From the examples directory, try

```
python SIR_enkf1.py
```

Directory Structure:
--------------------

	  pyda1.0/
		MANIFEST.in
		LICENSE
		README
		setup.py
		ez_setup.py
		examples/
			SIR_enkf1.py
			:
			data/
				:
			figures/
				:

		pyda/
			__init__.py
			analysis_generator/
				__init__.py
				analysis_generator_class.py
				:
				kf/
					__init__.py
					enkf1.py
					:
				pf/
					__init__.py
					:
			assimilation/
				__init__.py
				assimilation_current.py
				data_assimilation_class.py
				:
			ensemble_generator/
				__init__.py
				ensemble_generator_class.py
				SIRensemble.py
				:
			utilities/
				__init__.py
				epiODElib.py
				:
		
Description:
------------

examples directory:
	 
	 SIR_enkf1.py, an implementation of ensemble Kalman Filter
	 data assimilation using pyda. This serves as a basic example
	 of how the pyda classes are used together with a simulation.

pyda directory:
	
	This contains class files and helper module files in the
	utilities directory.

analysis_generator:

	Here classes are defined to implement data assimilation
	filters. The analysis classes are meant to be derived all from
	the AnalysisGeneratorClass defined in
	analysis_generator_class.py. Data assimilation filters are
	divided into Kalman Filter type and Particle Filter type.

kf:

	An example of an Ensemble Kalman filter analysis class is
	defined in enkf1.py. Other variants of Kalman filters are to
	be included here.

pf:

	Variants of particle filter and sequential monte carlo
	analysis schemes are meant to be included here.

assimilation:

	Beyond generating ensembles and producing filtered analysis
	ensembles data assimilation must control exactly how these
	schemes interact with data. In this directory classes that
	handle this are included. All data assimilation classes are
	meant to be derived from the DataAssimilationClass defined in
	data_assimilation_class.py. DA_current defined in
	assimilation_current.py is an example of such a class.

ensemble_generator:

	The user must already have software for simulation in place to
	use pyda. If this is the case then an ensemble generator class
	will control how the simulation produces forecasts. Classes
	are all derived from EnsembleGeneratorClass which is defined
	in ensemble_generator_class.py. SIRensemble.py defines a
	specific class to call on SIR epidemic simulation defined in
	the epiODElib.py module under utilities/.

utilities:

	This is a directory of helper modules. Currently this contains
	epiODEli.py which implements Runge-Kutta solvers for several
	differential equation based epidemic models. This directory
	also contains a basic visualization module AssimilationVis.py
	which uses matplotlib to visualize ensemble and analysis
	solutions.