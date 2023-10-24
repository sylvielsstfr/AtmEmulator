# Simple atmospheric transparency Emulator
#
#- author : Sylvie Dagoret-Campagne
#- affiliation : IJCLab/IN2P3/CNRS
#- creation date : 2023/10/24
#- last update : 2023/10/24
#This emulator is based from datagrid of atmospheric transparencies extracted from libradtran

import os
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pickle
from importlib.resources import files

from simpleemulator.simpleatmemulator import SimpleAtmEmulator,find_data_path,final_path_data
from simpleemulator.simpleatmemulator import Dict_Of_sitesAltitudes,Dict_Of_sitesPressures



class AtmPressEmulator(SimpleAtmEmulator):
    """
    Emulate Atmospheric Transparency above LSST from a data grids
    extracted from libradtran and analytical functions for aerosols.
    There are 3 grids:
    - 2D grid Rayleigh transmission vs (wavelength,airmass)
    - 2D grid O2 absorption vs  (wavelength,airmass)
    - 3D grid for PWV absorption vs (wavelength,airmass,PWV)
    - 3D grid for Ozone absorption vs (wavelength,airmass,Ozone)
    - Aerosol transmission for any number of components

    It uses the SimpleAtm Emulator. This particular class interpolate transparency
    with local pressures.
    """
    def __init__(self,obs_str = "LSST", pressure = 0 , path = final_path_data) : 
        SimpleAtmEmulator.__init__(self,obs_str = obs_str, path=path)
        """
        Initialize the class for data point files from which the 2D and 3D grids are created.
        Interpolation are calculated from the scipy RegularGridInterpolator() function
        Both types of data : trainging data for normal interpolaton use and the test data used
        to check accuracy of the interpolation of data.

        parameters :
          obs_str : pre-defined observation site tag corresponding to data files in data path
          pressure : pressure for which one want the transmission in mbar or hPa
          path    : path for data files
        """
    
        self.pressure = pressure
        self.refpressure = Dict_Of_sitesPressures[obs_str]
        self.pressureratio = self.pressure/self.refpressure
        if pressure == 0.0:
            self.pressureratio = 1



    def GetRayleighTransparencyArray(self,wl,am):
        return np.power(super().GetRayleighTransparencyArray(wl,am),self.pressureratio)
    

    def GetO2absTransparencyArray(self,wl,am):
        return np.power(super().GetO2absTransparencyArray(wl,am),self.pressureratio)
        
    
 