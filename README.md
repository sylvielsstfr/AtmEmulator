# Package AtmEmulator

Purpose : Python Emulators of libradtran atomospheric transparency for Rubin-LSST observatory
- Author : Sylvie Dagoret-Campagne
- Affiliation : IJCLab/IN2P3/CNRS
- date : February 2023


This is a python package dedicated to fast emulators of atmospheric transparency at the Rubin LSST observatory.

Grids of those transparencies are extracted from libRadtran - a library for radiative transfer - providing programs for calculation of solar and thermal radiation in the Earth's atmosphere.
Please refer to  http://www.libradtran.org for more information.

## Installation
(refer to installation procedure at https://setuptools.pypa.io/en/stable/userguide/quickstart.html)

After cloning the repo from github,
the installation may proceed as follow:

         go in the package top directory (where the pyproject.toml file is):
         cd AtmEmulator

         python -m pip install -e .
         or
         python -m pip install .



## Usage 


            from atmosphtransmemullsst.simpleatmospherictransparencyemulator import SimpleAtmEmulator

            emul = SimpleAtmEmulator()

            # definitions of atmospheric parameters
            WL = np.linspace(350,1100, 100)  # wavelengths array
            am = 1.2  # airmass
            pwv = 5.0 # precipitable water vapor vertical column depth in mm
            oz = 400. # ozone vertical column depth in Dobson Unit     
            ncomp=1  # number of aerosol components :
            taus= [0.02] # vertical aerosol optical depth at 550 nm (one per component)
            betas = [-1] # Angtrom exponent (one per component)

          

            transm = emul.GetAllTransparencies(WL,am,pwv0,oz0,ncomp=ncomp, taus=taus, betas=betas, flagAerosols=True)

            transm is the array of  the atmospheric transmission (one element per wavelength element)


            for more details, refer to the notebooks.

      

