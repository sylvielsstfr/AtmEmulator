Data grid and Interpolation 
============================
      

The atmospheric transparencies were sampled from libradtran according a regular grid
in wavelength, airmass, precipitable water vapor and Ozone.


To avoid a 4-D grid sampling, the light-air interaction processes were separated
independently to reduce the dimension of the grid sampling.

The different following interaction processes sampled according a grid:

* the Rayleigh scattering, 

* the Oxygen absorption process, 

* the Precipitable water vapor (pwv) absorption process, 

* the Ozone (oz) absorption process,

The base class ``SimpleAtmEmulator`` implements the interpolation functions which are
accessible to the user end-point or user interface class ``AtmEmulator``.


==================== ============ ===================== ========================================= 
**Interac process**  **grid-dim**  **data grid params**       **interpolation function**  
-------------------- ------------ --------------------- ----------------------------------------- 
 Rayleigh scattering     2D          (wl,airmass)       ``AtmEmulator.GetRayleighTransparencyArray``
 O2 absorption           2D          (wl,airmass)       ``AtmEmulator.GetO2absTransparencyArray``
 PWV absortion           3D        (wl,airmass,pwv)     ``AtmEmulator.GetPWVabsTransparencyArray``
 OZ absorption           3D        (wl,airmass,oz)      ``AtmEmulator.GetOZabsTransparencyArray``
==================== ============ ===================== ========================================= 

*where wl means wavelength*.

The range in parameters is limitted to the range of parameters of the datagrid.
To check the range of the parameters suitable for the emulator, please do the following
check:


.. code::

   >>> from atmemulator.atmemulator import AtmEmulator
   >>> emul =  AtmEmulator()
   >>> # or
   >>> emul =  AtmEmulator('CTIO')
   >>> # or 
   >>> emul =  AtmEmulator('LSST',743.0)
   
   >>> # check the wavelength range

   >>> emul.WL[0:10]
   array([300.        , 301.00125156, 302.00250313, 303.00375469,
       304.00500626, 305.00625782, 306.00750939, 307.00876095,
       308.01001252, 309.01126408])
   >>> emul.WL[-10:]
   array([1090.98873592, 1091.98998748, 1092.99123905, 1093.99249061,
       1094.99374218, 1095.99499374, 1096.99624531, 1097.99749687,
       1098.99874844, 1100.        ])


   >>> # check the airmass range
   >>> emul.AIRMASS
   array([1. , 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2,
       2.3, 2.4, 2.5])

   >>> # check the pwv range    
   >>> emul.PWV
   array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ,
        2.25,  2.5 ,  2.75,  3.  ,  3.25,  3.5 ,  3.75,  4.  ,  4.25,
        4.5 ,  4.75,  5.  ,  5.25,  5.5 ,  5.75,  6.  ,  6.25,  6.5 ,
        6.75,  7.  ,  7.25,  7.5 ,  7.75,  8.  ,  8.25,  8.5 ,  8.75,
        9.  ,  9.25,  9.5 ,  9.75, 10.  , 10.25, 10.5 , 10.75])

   >>> #check the ozone range

   >>> emul.OZ
   rray([  0,  25,  50,  75, 100, 125, 150, 175, 200, 225, 250, 275, 300,
       325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575])


Depending on the version of this ``AtmEmulator`` package, those grid size and sampling may depends
on the observation site.

