# atmospheric transparency Emulator
#
#- author : Sylvie Dagoret-Campagne
#- affiliation : IJCLab/IN2P3/CNRS
#- creation date : 2023/10/24
#- last update : 2023/10/24
#This emulator is based from datagrid of atmospheric transparencies extracted from libradtran

from simpleemulator.simpleatmemulator import SimpleAtmEmulator,find_data_path,final_path_data
from simpleemulator.simpleatmemulator import Dict_Of_sitesAltitudes,Dict_Of_sitesPressures
from atmpressemulator.atmpressemulator import AtmPressEmulator


class AtmEmulator(AtmPressEmulator):
    """
    Emulate Atmospheric Transparency above different sites.
    The preselected sites are LSST,CTIO, Mauna Kea, Observatoire de Haute Provence,
    Pic du Midi or Sea Level.
    Each site corresponds has an corresponding pressure. If the pressure does not correspond
    to the standard one, the pressure can be renormalized.

    AtmEmulator is the user end-point which call the official implementation of the emulator.
    
    By now AtmEmulator refer to the AtmPressEmulator which itself rely on the SimpleAtmEmulator.
    """
    def __init__(self,obs_str = "LSST", pressure = 0 , path = final_path_data) : 
        AtmPressEmulator.__init__(self,obs_str = obs_str, pressure = pressure , path=path)
        """
        Initialize the AtmEmulator. 

        parameters :
          obs_str : pre-defined observation site tag corresponding to data files in data path
          pressure : pressure for which one want the transmission in mbar or hPa
          path    : path for data files
        """
        pass
