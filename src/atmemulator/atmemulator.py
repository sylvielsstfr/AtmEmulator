# atmospheric transparency Emulator
#
#- author : Sylvie Dagoret-Campagne
#- affiliation : IJCLab/IN2P3/CNRS
#- creation date : 2023/10/24
#- last update : 2023/10/24
#This emulator is based from datagrid of atmospheric transparencies extracted from libradtran

import sys,getopt
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
    print("===========================================================================")
    print(f"Atmospheric Emulator for {obs_str} observatory and pressure = {pressure:.2f} hPa") 
    print("===========================================================================")
    
    
    emul = AtmEmulator(obs_str = obs_str, pressure = pressure)
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

