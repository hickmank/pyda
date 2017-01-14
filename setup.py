import os

from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
	name = 'pyda',
	version = '1.0.1',
	description = "pyda: An object-oriented data assimilation package",
	author = 'Kyle Hickmann',
	author_email = 'hickmank@gmail.com',
	maintainer = 'Kyle Hickmann',
	maintainer_email = 'hickmank@gmail.com',
	url = 'https://hickmank.github.io/pyda',
        long_description = read('README.md'),
	license = 'Apache 2.0',
	classifiers = [
                'Development Status :: 3 - Alpha',
		'Intended Audience :: Developers',
		'Intended Audience :: Education',
		'Intended Audience :: Science/Research',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
                'Topic :: Scientific/Engineering :: Information Analysis',
                'Topic :: Scientific/Engineering :: Mathematics',
                'Topic :: Scientific/Engineering :: Physics',
		'Operating System :: Unix',
		'Programming Language :: Python :: 2.7',
                'License :: OSI Approved :: Apache Software License'
		],
	packages = [
		'pyda',
		],
)
