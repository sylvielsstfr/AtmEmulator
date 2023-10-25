# Simple atmospheric transparency Emulator
#
#- author : Sylvie Dagoret-Campagne
#- affiliation : IJCLab/IN2P3/CNRS
#- creation date : 2023/10/24
#- last update : 2023/10/25
#This emulator is based from datagrid of atmospheric transparencies extracted from libradtran

import numpy as np
import sys,getopt
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
        """
        Scaling of optical depth by the term P/Pref, where P is the true pressure
        and Pref is the reference pressure for the site.
        """
        return np.power(super().GetRayleighTransparencyArray(wl,am),self.pressureratio)
    

    def GetO2absTransparencyArray(self,wl,am,satpower=1.1):
        """
        Correction of O2 absorption profile by the P/Pref with a power estimated
        from libradtran simulations, where P is the true pressure
        and Pref is the reference pressure for the site.

        Comparing LSST site with pressure at Mauna Kea and Sea Level show the satpower
        = 1.1 is appropriate.
        """
        return np.power(super().GetO2absTransparencyArray(wl,am),
                        np.power(self.pressureratio,satpower))
    

def usage():
    print("*******************************************************************")
    print(sys.argv[0],' -s<observation site-string> -p pressure')
    print("Observation sites are : ")
    print(' '.join(Dict_Of_sitesAltitudes.keys()))
    print('if pressure is not given, the standard pressure for the site is used')


    print('\t actually provided : ')
    print('\t \t Number of arguments:', len(sys.argv), 'arguments.')
    print('\t \t Argument List:', str(sys.argv))


def run(obs_str, pressure):
    print("==============================================================================")
    print(f"Atmospheric Pressure Emulator for {obs_str} observatory and pressure = {pressure:.2f} hPa") 
    print("==============================================================================")
    
    
    emul = AtmPressEmulator(obs_str = obs_str, pressure = pressure)
    wl = [400.,800.,900.]
    am=1.2
    pwv =4.0
    oz=300.
    transm = emul.GetAllTransparencies(wl,am,pwv,oz)
    print("wavelengths (nm) \t = ",wl)
    print("transmissions    \t = ",transm)

def is_float(element: any) -> bool:
    #If you expect None to be passed:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False

if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:],"hs:p:",["s=","p="])
    except getopt.GetoptError:
        print(' Exception bad getopt with :: '+sys.argv[0]+ ' -s<observation-site-string>')
        sys.exit(2)

    print('opts = ',opts)
    print('args = ',args)

    obs_str = ""
    pressure_str =""

    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-s", "--site"):
            obs_str = arg
        elif opt in ("-p", "--pressure"):
            pressure_str = arg
       
    if is_float(pressure_str):
        pressure = float(pressure_str)
    elif pressure_str =="":
        pressure = 0
    else:
        print(f"Pressure argument {pressure_str} is not a float")
        sys.exit()

        
    if obs_str in Dict_Of_sitesAltitudes.keys():
        run(obs_str=obs_str,pressure = pressure)
    else:
        print(f"Observatory {obs_str} not in preselected observation site")
        print(f"This site {obs_str} must be added in libradtranpy preselected sites")
        sys.exit()

    
        
    
 