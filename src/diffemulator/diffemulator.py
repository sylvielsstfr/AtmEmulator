# Simple differentiable atmospheric transparency Emulator
#
#- author : Sylvie Dagoret-Campagne
#- affiliation : IJCLab/IN2P3/CNRS
#- creation date : 2023/10/25
#- last update : 2023/10/25
# This emulator is based from datagrid of atmospheric transparencies extracted from libradtran

import os
import sys,getopt
from pathlib import Path
import numpy as np
from interpolate import RegularGridInterpolator
import pickle


# preselected sites 
Dict_Of_sitesAltitudes = {'LSST':2.663,
                          'CTIO':2.207,
                          'OHP':0.65,
                          'PDM':2.8905,
                          'OMK':4.205,
                          'OSL':0.000,
                           }
# pressure calculated by libradtran
Dict_Of_sitesPressures = {'LSST':731.50433,
                          'CTIO':774.6052,
                          'OHP':937.22595,
                          'PDM':710.90637,
                          'OMK':600.17224,
                          'OSL':1013.000,
                        }



# about datapath

dir_path_data = os.path.join(os.path.dirname(__file__), 'data')

file_data_dict = {
    "info" :"atmospherictransparencygrid_params.pickle",
    "data_rayleigh" : "atmospherictransparencygrid_rayleigh.npy",
    "data_o2abs" : "atmospherictransparencygrid_O2abs.npy",
    "data_pwvabs": "atmospherictransparencygrid_PWVabs.npy",
    "data_ozabs" : "atmospherictransparencygrid_OZabs.npy",
}

def find_data_path():
    """
    Search the path for the atmospheric emulator.
    Normaly there is no need to debug where are the data given the dir_path_data is
    the good place to find the data. It is left for debug purpose
    """
    
    print("\t - dir_path_data ",dir_path_data)
    
    dir_file_abspath = os.path.dirname(os.path.abspath(__file__))
    print("dir_file_abspath = ",dir_file_abspath)

    dir_file_realpath = os.path.dirname(os.path.realpath(__file__))
    print("dir_file_realpath = ",dir_file_realpath)

    dir_file_sys = Path(sys.path[0])
    print("Path_syspath = ",dir_file_sys)

    dir_file_dirname = os.path.dirname(__file__) 
    print("file_dirname = ",dir_file_dirname)
    
    #path_data = dir_file_realpath + dir_path_data
    path_data = dir_path_data
    
    for key, filename in file_data_dict.items():
        file_path = os.path.join(path_data ,  filename)
        flag_found = os.path.isfile(file_path)  
        if flag_found :
            print(f"found data file {file_path}")
        else:
            print(f">>>>>>>>>> NOT found data file {file_path} in dir {path_data}")
            
    return path_data 

# inal_path_data is defined in case dir_path_data_data would not be the right data path 
final_path_data = dir_path_data



# preselected sites in libradtranpy

Dict_Of_sitesAltitudes = {'LSST':2.663,
                          'CTIO':2.207,
                          'OHP':0.65,
                          'PDM':2.8905,
                          'OMK':4.205,
                          'OSL':0,
                           }


class SimpleDiffAtmEmulator:
    """
    Emulate Atmospheric Transparency above LSST from a data grids
    extracted from libradtran and analytical functions for aerosols.
    There are 3 grids:
    - 2D grid Rayleigh transmission vs (wavelength,airmass)
    - 2D grid O2 absorption vs  (wavelength,airmass)
    - 3D grid for PWV absorption vs (wavelength,airmass,PWV)
    - 3D grid for Ozone absorption vs (wavelength,airmass,Ozone)
    - Aerosol transmission for any number of components
    
    """
    def __init__(self,obs_str = "LSST", path = final_path_data) :
        """
        Initialize the class for data point files from which the 2D and 3D grids are created.
        Interpolation are calculated from the scipy RegularGridInterpolator() function
        Both types of data : trainging data for normal interpolaton use and the test data used
        to check accuracy of the interpolation of data.

        parameters :
          obs_str : pre-defined observation site tag corresponding to data files in data path
          path    : path for data files
        """
        OBS_tag = ""
        if obs_str in Dict_Of_sitesAltitudes.keys():
            OBS_tag = obs_str
            print(f"Observatory {obs_str} found in preselected observation sites")
        else:
            print(f"Observatory {obs_str} not in preselected observation sites")
            print(f"This site {obs_str} must be added in libradtranpy preselected sites")
            print(f"and generate corresponding scattering and absorption profiles")
            sys.exit()

        # construct the path of input data files
        self.path = path
        self.fn_info = OBS_tag + "_" + file_data_dict["info"]
        self.fn_rayleigh = OBS_tag + "_" + file_data_dict["data_rayleigh"]
        self.fn_O2abs = OBS_tag + "_" + file_data_dict["data_o2abs"]
        self.fn_PWVabs = OBS_tag + "_" + file_data_dict["data_pwvabs"]
        self.fn_OZabs = OBS_tag + "_" + file_data_dict["data_ozabs"]
       
        self.info_params = None
        self.data_rayleigh = None
        self.data_O2abs = None
        self.data_PWVabs = None
        self.data_OZabs = None
     
        
        # load all data files (training and test)
        self.loadtables()
        
        # setup training dataset (those used for interpolation)
        self.WLMIN = self.info_params["WLMIN"]
        self.WLMAX = self.info_params["WLMAX"]
        self.WLBIN = self.info_params["WLBIN"]
        self.NWLBIN = self.info_params['NWLBIN']
        self.WL = self.info_params['WL']
        self.OBS = self.info_params['OBS']
        
        self.AIRMASSMIN = self.info_params['AIRMASSMIN']
        self.AIRMASSMAX = self.info_params['AIRMASSMAX']
        self.NAIRMASS = self.info_params['NAIRMASS']
        self.DAIRMASS = self.info_params['DAIRMASS']
        self.AIRMASS = self.info_params['AIRMASS']
        
        self.PWVMIN = self.info_params['PWVMIN']
        self.PWVMAX = self.info_params['PWVMAX'] 
        self.NPWV = self.info_params['NPWV']
        self.DPWV = self.info_params['DPWV'] 
        self.PWV = self.info_params['PWV']
        
        
        self.OZMIN =  self.info_params['OZMIN']
        self.OZMAX = self.info_params['OZMAX']
        self.NOZ = self.info_params['NOZ']
        self.DOZ =  self.info_params['DOZ'] 
        self.OZ = self.info_params['OZ']
        
       


        # constant parameters defined for aerosol formula
        self.lambda0 = 550.
        self.tau0 = 1.

        # interpolation is done over training dataset
        self.func_rayleigh = RegularGridInterpolator((self.WL,self.AIRMASS),self.data_rayleigh)
        self.func_O2abs = RegularGridInterpolator((self.WL,self.AIRMASS),self.data_O2abs)
        self.func_PWVabs = RegularGridInterpolator((self.WL,self.AIRMASS,self.PWV),self.data_PWVabs)
        self.func_OZabs = RegularGridInterpolator((self.WL,self.AIRMASS,self.OZ),self.data_OZabs)

        
        
    def loadtables(self):
        """
        Load files into grid arrays
        """
        
        filename=os.path.join(self.path,self.fn_info)     
        with open(filename, 'rb') as f:
            self.info_params = pickle.load(f)
            
       
        filename=os.path.join(self.path,self.fn_rayleigh)
        with open(filename, 'rb') as f:
            self.data_rayleigh = np.load(f)
            
            
        filename=os.path.join(self.path,self.fn_O2abs)
        with open(filename, 'rb') as f:
            self.data_O2abs = np.load(f)
            
      
        filename=os.path.join(self.path,self.fn_PWVabs)
        with open(filename, 'rb') as f:
            self.data_PWVabs = np.load(f)
            
        
            
        filename=os.path.join(self.path,self.fn_OZabs)
        with open(filename, 'rb') as f:
            self.data_OZabs = np.load(f)
            
      
            
    # functions to access to interpolated transparency functions on training dataset
    #        
    def GetWL(self):
        return self.WL
            
    def GetRayleighTransparencyArray(self,wl,am):
        pts = [ (the_wl,am) for the_wl in wl ]
        pts = np.array(pts)
        return self.func_rayleigh(pts)
    
    
    def GetO2absTransparencyArray(self,wl,am):
        pts = [ (the_wl,am) for the_wl in wl ]
        pts = np.array(pts)
        return self.func_O2abs(pts)
    

    
    def GetPWVabsTransparencyArray(self,wl,am,pwv):
        pts = [ (the_wl,am,pwv) for the_wl in wl ]
        pts = np.array(pts)
        return self.func_PWVabs(pts)
    
    
    def GetOZabsTransparencyArray(self,wl,am,oz):
        pts = [ (the_wl,am,oz) for the_wl in wl ]
        pts = np.array(pts)
        return self.func_OZabs(pts)
    
        
    def GetGriddedTransparencies(self,wl,am,pwv,oz,flagRayleigh=True,flagO2abs=True,flagPWVabs=True,flagOZabs=True):
        """
        Emulation of libradtran simulated transparencies. Decomposition of the
        total transmission in different processes:
        - Rayleigh scattering
        - O2 absorption
        - PWV absorption
        - Ozone absorption
        
        inputs:
        - wl : wavelength array or list
        - am :the airmass,
        - pwv : the precipitable water vapor (mm)
        - oz : the ozone column depth in Dobson unit
        - flags to activate or not the individual interaction processes
        
        outputs:
        - 1D array of atmospheric transmission (save size as wl)
        
        """
        

        if flagRayleigh:
            transm = self.GetRayleighTransparencyArray(wl,am)
        else:
            transm = np.ones(len(wl))
            
        if flagO2abs:
            transm *= self.GetO2absTransparencyArray(wl,am)
            
        if flagPWVabs:
            transm *= self.GetPWVabsTransparencyArray(wl,am,pwv)
            
        if flagOZabs:
            transm *= self.GetOZabsTransparencyArray(wl,am,oz)
            
        return transm
            
    def GetAerosolsTransparencies(self,wl,am,ncomp,taus=None,betas=None):
        """
        Compute transmission due to aerosols:
        
        inputs:
        - wl : wavelength array
        - am : the airmass
        - ncomp : the number of aerosol components
        - taus : the vertical aerosol depth of each component at lambda0 vavelength
        - betas : the angstrom exponent. Must be negativ.
        
        
        outputs:
        - 1D array of atmospheric transmission (save size as wl)
        
        """
          
        wl = np.array(wl)
        NWL=wl.shape[0]
        
        transm = np.ones(NWL)
        
        if ncomp <=0:
            return transm
        else:
            taus=np.array(taus)
            betas=np.array(betas)
            
            NTAUS=taus.shape[0]
            NBETAS=betas.shape[0]
        
            assert ncomp<=NTAUS
            assert ncomp<=NBETAS     
        
            for icomp in range(ncomp):            
                exponent = (taus[icomp]/self.tau0)*np.exp(betas[icomp]*np.log(wl/self.lambda0))*am
                transm *= np.exp(-exponent)
            
            return transm
        
        
    def GetAllTransparencies(self,wl,am,pwv,oz,ncomp=0, taus=None, betas=None, flagRayleigh=True,flagO2abs=True,flagPWVabs=True,flagOZabs=True,flagAerosols=False):
        """
        Combine interpolated libradtran transmission with analytical expression for the
        aerosols
        
        inputs:
        - wl : wavelength array or list
        - am :the airmass,
        - pwv : the precipitable water vapor (mm)
        - oz : the ozone column depth in Dobson unit
        - ncomp : number of aerosols components,
        - taus & betas : arrays of parameters for aerosols
        - flags to activate or not the individual interaction processes
        
        outputs:
        - 1D array of atmospheric transmission (save size as wl)
        
        """
        
        transm = self.GetGriddedTransparencies(wl,am,pwv,oz,flagRayleigh=flagRayleigh,flagO2abs=flagO2abs,flagPWVabs=flagPWVabs,flagOZabs=flagOZabs)
        
        if flagAerosols:
            transmaer = self.GetAerosolsTransparencies(wl,am,ncomp,taus,betas)
            transm *=transmaer
           
            
        return transm
    


def usage():
    print("*******************************************************************")
    print(sys.argv[0],' -s<observation site-string>')
    print("Observation sites are : ")
    print(' '.join(Dict_Of_sitesAltitudes.keys()))


    print('\t actually provided : ')
    print('\t \t Number of arguments:', len(sys.argv), 'arguments.')
    print('\t \t Argument List:', str(sys.argv))




def run(obs_str):
    print("============================================================")
    print(f"Simple Differentiable Atmospheric emulator for {obs_str} observatory")
    print("============================================================")
    
    # retrieve the path of data
    #path_data =  find_data_path()  
    # create emulator  
    #emul = SimpleAtmEmulator(path = path_data)
    emul = SimpleDiffAtmEmulator(obs_str=obs_str)
    wl = [400.,800.,900.]
    am=1.2
    pwv =4.0
    oz=300.
    transm = emul.GetAllTransparencies(wl,am,pwv,oz)
    print("wavelengths (nm) \t = ",wl)
    print("transmissions    \t = ",transm)
    
    

if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:],"hs:",["s="])
    except getopt.GetoptError:
        print(' Exception bad getopt with :: '+sys.argv[0]+ ' -s<observation-site-string>')
        sys.exit(2)

    print('opts = ',opts)
    print('args = ',args)

    obs_str = ""
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-s", "--site"):
            obs_str = arg
       

        
    if obs_str in Dict_Of_sitesAltitudes.keys():
        run(obs_str=obs_str)
    else:
        print(f"Observatory {obs_str} not in preselected observation site")
        print(f"This site {obs_str} must be added in libradtranpy preselected sites")
        sys.exit()


    
            
            