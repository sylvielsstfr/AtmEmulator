# Simple differentiable atmospheric transparency Emulator
#
#- author : Sylvie Dagoret-Campagne
#- affiliation : IJCLab/IN2P3/CNRS
#- creation date : 2023/10/25
#- last update : 2023/10/26
# This emulator is based from datagrid of atmospheric transparencies extracted from libradtran

import os
import sys,getopt
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap,jacobian,jacfwd, hessian,lax
import jax.numpy as jnp
from functools import partial
from diffemulator.interpolate import RegularGridInterpolator
#from interpolate import RegularGridInterpolator
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
final_path_data = dir_path_data

file_data_dict = {
    "info" :"atmospherictransparencygrid_params.pickle",
    "data_rayleigh" : "atmospherictransparencygrid_rayleigh.npy",
    "data_o2abs" : "atmospherictransparencygrid_O2abs.npy",
    "data_pwvabs": "atmospherictransparencygrid_PWVabs.npy",
    "data_ozabs" : "atmospherictransparencygrid_OZabs.npy",
}

# inal_path_data is defined in case dir_path_data_data would not be the right data path 



# preselected sites in libradtranpy

Dict_Of_sitesAltitudes = {'LSST':2.663,
                          'CTIO':2.207,
                          'OHP':0.65,
                          'PDM':2.8905,
                          'OMK':4.205,
                          'OSL':0,
                           }


 

@partial(jit, static_argnums=2)
def _interpolatedfunctwoargs(x1,x2,func):
    pts = jnp.meshgrid(x1,x2)
    pts_stacked = jnp.dstack(pts)
    return func(pts_stacked)

@partial(jit, static_argnums=2)
def _interpolatedfunctwoargs_scalar(x1,x2,func):
    pts = (x1,x2)
    pts_stacked = jnp.array([pts])
    return func(pts_stacked)[0]


# specify arguments https://stackoverflow.com/questions/66445754/understanding-jax-argnums-parameter-in-its-gradient-function
_jac__interpolatedfunctwoargs = jacfwd(_interpolatedfunctwoargs,argnums=(0, 1))

# all differentiations wrt a scalar function of two arguments
_dfdx_xy = grad(_interpolatedfunctwoargs_scalar,argnums=0)
_dfdy_xy = grad(_interpolatedfunctwoargs_scalar,argnums=1)
_d2fdxx_xy = grad(_dfdx_xy,argnums=0)
_d2fdxy_xy = grad(_dfdx_xy,argnums=1)
_d2fdyx_xy = grad(_dfdy_xy,argnums=0)
_d2fdyy_xy = grad(_dfdy_xy,argnums=1)


@partial(jit, static_argnums=3)
def _interpolatedfuncthreeargs_scalar(x1,x2,x3,func):
    pts = (x1,x2,x3)
    pts_stacked = jnp.dstack([pts])
    return func(pts)[0]

# all differentiations wrt a scalar function of two arguments
_dfdx_xyz = grad(_interpolatedfuncthreeargs_scalar,argnums=0)
_dfdy_xyz = grad(_interpolatedfuncthreeargs_scalar,argnums=1)
_dfdz_xyz = grad(_interpolatedfuncthreeargs_scalar,argnums=2)
_d2fdxx_xyz = grad(_dfdx_xyz,argnums=0)
_d2fdxy_xyz = grad(_dfdx_xyz,argnums=1)
_d2fdxz_xyz = grad(_dfdx_xyz,argnums=2)
_d2fdyx_xyz = grad(_dfdy_xyz,argnums=0)
_d2fdyy_xyz = grad(_dfdy_xyz,argnums=1)
_d2fdyz_xyz = grad(_dfdy_xyz,argnums=2)
_d2fdzx_xyz = grad(_dfdz_xyz,argnums=0)
_d2fdzy_xyz = grad(_dfdz_xyz,argnums=1)
_d2fdzz_xyz = grad(_dfdz_xyz,argnums=2)


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
        self.WL = jnp.array(self.WL)
        self.OBS = self.info_params['OBS']
        
        self.AIRMASSMIN = self.info_params['AIRMASSMIN']
        self.AIRMASSMAX = self.info_params['AIRMASSMAX']
        self.NAIRMASS = self.info_params['NAIRMASS']
        self.DAIRMASS = self.info_params['DAIRMASS']
        self.AIRMASS = self.info_params['AIRMASS']
        self.AIRMASS = jnp.array(self.AIRMASS)
        
        self.PWVMIN = self.info_params['PWVMIN']
        self.PWVMAX = self.info_params['PWVMAX'] 
        self.NPWV = self.info_params['NPWV']
        self.DPWV = self.info_params['DPWV'] 
        self.PWV = self.info_params['PWV']
        self.PWV = jnp.array(self.PWV)
        
        
        self.OZMIN =  self.info_params['OZMIN']
        self.OZMAX = self.info_params['OZMAX']
        self.NOZ = self.info_params['NOZ']
        self.DOZ =  self.info_params['DOZ'] 
        self.OZ = self.info_params['OZ']
        self.OZ = jnp.array(self.OZ)
        

        # constant parameters defined for aerosol formula
        self.lambda0 = 550.
        self.tau0 = 1.

        # interpolation functions are build on the loaded dataset
        self.func_rayleigh = RegularGridInterpolator((self.WL,self.AIRMASS),self.data_rayleigh)
        self.func_O2abs = RegularGridInterpolator((self.WL,self.AIRMASS),self.data_O2abs)
        self.func_PWVabs = RegularGridInterpolator((self.WL,self.AIRMASS,self.PWV),self.data_PWVabs)
        self.func_OZabs = RegularGridInterpolator((self.WL,self.AIRMASS,self.OZ),self.data_OZabs)
    
        # initialise flag that activate
        self.init_flagprocesses()


    def init_flagprocesses(self):
        self.dict_flags = dict(flagRayleigh=True,flagO2abs=True,flagPWVabs=True,flagOZabs=True,flagAerosols=True)
        self.put_individualflags()

    def put_individualflags(self):    
        self.flagRayleigh = self.dict_flags['flagRayleigh']
        self.flagO2abs = self.dict_flags['flagO2abs']
        self.flagPWVabs = self.dict_flags['flagPWVabs']
        self.flagOZabs = self.dict_flags['flagOZabs']
        self.flagAerosols = self.dict_flags['flagAerosols']

    def get_flagprocesses(self):
        return self.dict_flags
    
    def set_flagprocesses(self,the_dict):
        self.dict_flags = the_dict
        self.put_individualflags()


    def loadtables(self):
        """
        Load files into grid arrays.
        The data to be interpolated are converted in jax arrays
        """
        
        filename=os.path.join(self.path,self.fn_info)     
        with open(filename, 'rb') as f:
            self.info_params = pickle.load(f)
            
       
        filename=os.path.join(self.path,self.fn_rayleigh)
        with open(filename, 'rb') as f:
            self.data_rayleigh = jnp.load(f)
            #self.data_rayleigh = jnp.array(self.data_rayleigh)
            
            
        filename=os.path.join(self.path,self.fn_O2abs)
        with open(filename, 'rb') as f:
            self.data_O2abs = jnp.load(f)
            #self.data_O2abs = jnp.array(self.data_O2abs)
            
      
        filename=os.path.join(self.path,self.fn_PWVabs)
        with open(filename, 'rb') as f:
            self.data_PWVabs = jnp.load(f)
            #self.data_PWVabs = jnp.array(self.data_PWVabs)
            
        
            
        filename=os.path.join(self.path,self.fn_OZabs)
        with open(filename, 'rb') as f:
            self.data_OZabs = jnp.load(f)
            #self.data_OZabs = jnp.array(self.data_OZabs)
            
      
            
    # functions to access to interpolated transparency functions on training dataset
    #        
    def GetWL(self):
        return self.WL
    
    def GetRayleighTransparencyScalar(self,wl,am):
        pts = jnp.array([(wl,am)])
        return self.func_rayleigh(pts)[0]
            
    def GetRayleighTransparency1DArray(self,wl,am):
        pts = [ (the_wl,am) for the_wl in wl ]
        pts_stacked = jnp.array(pts)
        return self.func_rayleigh(pts_stacked)
    
    def GetO2absTransparencyScalar(self,wl,am):
        pts = jnp.array([(wl,am)])
        return self.func_O2abs(pts)[0]
    
    def GetO2absTransparency1DArray(self,wl,am):
        pts = [ (the_wl,am) for the_wl in wl ]
        pts_stacked = jnp.array(pts)
        return self.func_O2abs(pts_stacked)
    
    def GetPWVabsTransparencyScalar(self,wl,am,pwv):
        pts = jnp.array([ (wl,am,pwv) ])
        return self.func_PWVabs(pts)[0]
    
    def GetPWVabsTransparency1DArray(self,wl,am,pwv):
        pts = [ (the_wl,am,pwv) for the_wl in wl ]
        pts_stacked = jnp.array(pts)
        return self.func_PWVabs(pts_stacked)
    
    def GetOZabsTransparencyScalar(self,wl,am,oz):
        pts = jnp.array([(wl,am,oz)])
        return self.func_OZabs(pts)[0]
    
    def GetOZabsTransparency1DArray(self,wl,am,oz):
        pts = [ (the_wl,am,oz) for the_wl in wl ]
        pts_stacked = jnp.array(pts)
        return self.func_OZabs(pts_stacked)
    

    def GetGriddedTransparenciesScalar(self,wl,am,pwv,oz):
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
        -  Transmission at a given point (wl,am,pwv,oz)
        
        """
        
        
        full_transm1 = lambda   x , y : 1.0
        full_transm2  = lambda x , y , z : 1.0

        transm = lax.cond(self.flagRayleigh,self.GetRayleighTransparencyScalar,full_transm1,wl,am)
        transm *= lax.cond(self.flagO2abs,self.GetO2absTransparencyScalar,full_transm1,wl,am)
        transm *= lax.cond(self.flagPWVabs,self.GetPWVabsTransparencyScalar,full_transm2,wl,am,pwv)
        transm *= lax.cond(self.flagOZabs,self.GetOZabsTransparencyScalar,full_transm2,wl,am,oz)    
        return transm

    def GetGriddedTransparencies1DArray(self,wl,am,pwv,oz):
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


        full_transm1 = lambda   x , y : jnp.ones(len(wl)) 
        full_transm2  = lambda x , y , z : jnp.ones(len(wl)) 


        transm = lax.cond(self.flagRayleigh,self.GetRayleighTransparency1DArray,full_transm1,wl,am)
        transm *= lax.cond(self.flagO2abs,self.GetO2absTransparency1DArray,full_transm1,wl,am)
        transm *= lax.cond(self.flagPWVabs,self.GetPWVabsTransparency1DArray,full_transm2,wl,am,pwv)
        transm *= lax.cond(self.flagOZabs,self.GetOZabsTransparency1DArray,full_transm2,wl,am,oz)    
        return transm

    

    def GetAerosolsTransparenciesScalar(self,wl,am,tau=0,beta=-1):
        """
        Compute transmission due to aerosols:
        
        inputs:
        - wl : wavelength array
        - am : the airmass
        - tau : the vertical aerosol depth of each component at lambda0 vavelength
        - beta : the angstrom exponent. Must be negativ.
        
        
        outputs:
        - Value of the transmission
        
        """
              
        exponent = (tau/self.tau0)*jnp.exp(beta*jnp.log(wl/self.lambda0))*am
        transm = jnp.exp(-exponent)
            
        return transm
            
    def GetAerosolsTransparencies1DArray(self,wl,am,tau,beta):
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
          
        wl = jnp.array(wl)
        NWL=wl.shape[0]
        
                   
        exponent = (tau/self.tau0)*jnp.exp(beta*jnp.log(wl/self.lambda0))*am
        transm = jnp.exp(-exponent)
            
        return transm
        
        
    def GetAllTransparenciesScalar(self,wl,am,pwv,oz,tau=0, beta=-1): 
                                   
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

        
        full_transm  = lambda x , y , z ,t : 1.0
        
        transm = self.GetGriddedTransparenciesScalar(wl,am,pwv,oz)
        transm *=lax.cond(self.flagAerosols,self.GetAerosolsTransparenciesScalar,full_transm ,wl,am,tau,beta)  
        return transm
    

    def GetAllTransparencies1DArray(self,wl,am,pwv,oz,tau=0, beta=-1):
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
        
        full_transm  = lambda x , y , z ,t : jnp.ones(len(wl))  

        transm = self.GetGriddedTransparencies1DArray(wl,am,pwv,oz)
        transmaer = lax.cond(self.flagAerosols,self.GetAerosolsTransparencies1DArray,full_transm,wl,am,tau,beta)
        transm *=transmaer
        return transm

    # vectorize scalar functions along wl axis
    def vect1d_Rayleightransparency(self,wl,am):
        return jit(vmap(self.GetRayleighTransparencyScalar, in_axes=(0, None)))(wl,am)
    
    def vect1d_O2abstransparency(self,wl,am):
        return jit(vmap(self.GetO2absTransparencyScalar,in_axes=(0, None)))(wl,am)
    
    def vect1d_PWVabstransparency(self,wl,am,pwv):
        return jit(vmap(self.GetPWVabsTransparencyScalar,in_axes=(0,None,None)))(wl,am,pwv)

    def vect1d_OZabstransparency(self,wl,am,oz):
         return jit(vmap(self.GetOZabsTransparencyScalar,in_axes=(0,None,None)))(wl,am,oz)
    
    def vect1d_Griddedtransparency(self,wl,am,pwv,oz):
        return jit(vmap(self.GetGriddedTransparenciesScalar,in_axes=(0,None,None,None)))(wl,am,pwv,oz)


    def vect1d_Aerosolstransparency(self,wl,am,tau=0,beta=-1):
        return jit(vmap(self.GetAerosolsTransparenciesScalar,in_axes=(0,None,None,None)))(wl,am,tau,beta)

    def vect1d_Alltransparencies(self,wl,am,pwv,oz,tau=0,beta=-1):                   
        return vmap(self.GetAllTransparenciesScalar,
                        in_axes=(0,None,None,None,None,None))(wl,am,pwv,oz, tau, beta)
                                    
                                    
    # vectorize scalar functions along wl and airmass or another axis
    def vect2d_Rayleightransparency(self,wl,am):
        return jit(vmap(vmap(self.GetRayleighTransparencyScalar, in_axes=(0, None)),in_axes=(None,0)))(wl,am)

    def vect2d_O2abstransparency(self,wl,am):
        return jit(vmap(vmap(self.GetO2absTransparencyScalar,in_axes=(0, None)),in_axes=(None,0)))(wl,am)

    def vect2da_PWVabstransparency(self,wl,am,pwv):
        return jit(vmap(vmap(self.GetPWVabsTransparencyScalar,in_axes=(0, None,None)),in_axes=(None,0,None)))(wl,am,pwv)

    def vect2db_PWVabstransparency(self,wl,am,pwv):
        return jit(vmap(vmap(self.GetPWVabsTransparencyScalar,in_axes=(0, None,None)),in_axes=(None,None,0)))(wl,am,pwv)


    def vect2da_OZabstransparency(self,wl,am,oz):
        return jit(vmap(vmap(self.GetOZabsTransparencyScalar,in_axes=(0, None,None)),in_axes=(None,0,None)))(wl,am,oz)

    def vect2db_OZabstransparency(self,wl,am,oz):
        return jit(vmap(vmap(self.GetOZabsTransparencyScalar,in_axes=(0, None,None)),in_axes=(None,None,0)))(wl,am,oz)

    def vect2da_Aerosolstransparency(self,wl,am,tau,beta):
        return jit(vmap(vmap(self.GetAerosolsTransparenciesScalar,in_axes=(0, None,None,None)),in_axes=(None,0,None,None)))(wl,am,tau,beta)

    def vect2db_Aerosolstransparency(self,wl,am,tau,beta):
        return jit(vmap(vmap(self.GetAerosolsTransparenciesScalar,in_axes=(0, None,None,None)),in_axes=(None,None,0,None)))(wl,am,tau,beta)

    def vect2dc_Aerosolstransparency(self,wl,am,tau,beta):
        return jit(vmap(vmap(self.GetAerosolsTransparenciesScalar,in_axes=(0, None,None,None)),in_axes=(None,None,None,0)))(wl,am,tau,beta)


    def vect2da_Griddedtransparency(self,wl,am,pwv,oz):
        return jit(vmap(vmap(self.GetGriddedTransparenciesScalar,in_axes=(0, None,None,None)),in_axes=(None,0,None,None)))(wl,am,pwv,oz)

    def vect2db_Griddedtransparency(self,wl,am,pwv,oz):
        return jit(vmap(vmap(self.GetGriddedTransparenciesScalar,in_axes=(0, None,None,None)),in_axes=(None,None,0,None)))(wl,am,pwv,oz)

    def vect2dc_Griddedtransparency(self,wl,am,pwv,oz):
        return jit(vmap(vmap(self.GetGriddedTransparenciesScalar,in_axes=(0, None,None,None)),in_axes=(None,None,None,0)))(wl,am,pwv,oz)



    def GetRayleighTransparency2D(self,wl,am):
        return _interpolatedfunctwoargs(wl,am,self.func_rayleigh)
    
    def GetO2absTransparency2D(self,wl,am):
        return _interpolatedfunctwoargs(wl,am,self.func_O2abs)
    
     
    def DiffGetRayleighTransparencyScalar(self,wl,am):
        return _dfdx_xy(wl,am,self.func_rayleigh)
    

    
   



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
    wl = jnp.array([400.,800.,900.])
    am=1.2
    pwv =4.0
    oz=300.
    transm = emul.GetAllTransparencies1DArray(wl,am,pwv,oz)
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


    
            
            