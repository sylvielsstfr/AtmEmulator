# atmemulator

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)



# Package AtmEmulator Description

Purpose : Python Emulators of libradtran atomospheric transparency for Rubin-LSST observatory
- Author : Sylvie Dagoret-Campagne
- Affiliation : IJCLab/IN2P3/CNRS
- date : October 2023


This is a python package dedicated to fast emulators of atmospheric transparency at the Rubin LSST observatory.

Grids of those transparencies are extracted from libRadtran - a library for radiative transfer - providing programs for calculation of solar and thermal radiation in the Earth's atmosphere.
Please refer to  http://www.libradtran.org for more information.

## Installation
(refer to installation procedure at https://setuptools.pypa.io/en/stable/userguide/quickstart.html)

After cloning the repo from github,
the installation may proceed as follow:

         go in the package top directory (where the pyproject.toml file is):
         cd ``AtmEmulator``

         pip install -e .'[dev]'

         or

         python install -e . [dev]



## Usage 

### Simple Usage

The Top class ``SimpleAtmEmulator`` based on an interpolator grid of atmospheric transmission can be called as follow from a python application :

   - choosing the observation site among a pre-defined list names 
   'LSST','CTIO','OHP' for observatoire de Haute Provence,'PDM' for observatoire du Pic du Midi,'OMK' for Mauna Kea,'OSL' for Sea Level
   
            # set the observation site
            obs_str = 'LSST'
            the_pressure = 743.0 # in mbar or hPa

         
            # import the module and the corresponding class
            from atmemulator.atmemulator import AtmEmulator
            from atmemulator.atmemulator import Dict_Of_sitesAltitudes,Dict_Of_sitesPressures

            # create the emulator
            # if you don't provide the pressure, the default pressure of the site given in
            ``Dict_Of_sitesPressures`` is used (the standard tabulated one).

            emul = SimpleAtmEmulator(obs_str = obs_str,pressure = the_pressure)

            # definitions of atmospheric parameters
            WL = np.linspace(350,1100, 100)  # wavelengths array
            am = 1.2  # airmass
            pwv = 5.0 # precipitable water vapor vertical column depth in mm
            oz = 400. # ozone vertical column depth in Dobson Unit     
            ncomp=1  # number of aerosol components :
            taus= [0.02] # vertical aerosol optical depth at 550 nm (one per component)
            betas = [-1] # Angtrom exponent (one per component)

          
            # compute the transmission
            transm = emul.GetAllTransparencies(WL,am,pwv0,oz0,ncomp=ncomp, taus=taus, betas=betas, flagAerosols=True)

            transm is the array of  the atmospheric transmission (one element per wavelength element)


   for more details, please refer to the [notebooks](docs/notebooks/intro_notebook.ipynb).


   It is also possible to use this emulator for other sites like CTIO, Mauna Kea, Observatoire de Haute Provence, Pic du Midi, Sea Level.

### More advanced usage

An avanced version based on Jax for autodifferentiation is implemented



## Tests

Please refer to [docs/notebooks](docs/notebooks/intro_notebook.ipynb) to run examples for using `simpleemulator`  including its accuracy tests.



## Reference to Python Project Template


This project was automatically generated using the LINCC-Frameworks 
[python-project-template](https://github.com/lincc-frameworks/python-project-template).

A repository badge was added to show that this project uses the python-project-template, however it's up to
you whether or not you'd like to display it!

For more information about the project template see the 
[documentation](https://lincc-ppt.readthedocs.io/en/latest/).

## Dev Guide - Getting Started

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

```
>> conda create env -n <env_name> python=3.10
>> conda activate <env_name>
```

Once you have created a new environment, you can install this project for local
development using the following commands:

```
>> pip install -e .'[dev]'
>> pre-commit install
>> conda install pandoc
```

Notes:
1) The single quotes around `'[dev]'` may not be required for your operating system.
2) `pre-commit install` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on 
   [pre-commit](https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html)
3) Install `pandoc` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   [Sphinx and Python Notebooks](https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks)





