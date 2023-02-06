#!/usr/bin/env python
# coding: utf-8

# # Generate grids


# Import some generally useful packages

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pandas as pd
from itertools import cycle, islice
import seaborn as sns




# to enlarge the sizes
params = {'legend.fontsize': 'large',
          'figure.figsize': (8, 6),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
plt.rcParams.update(params)


from scipy import interpolate



from libradtranpy import  libsimulateVisible



import warnings
warnings.filterwarnings('ignore')


# # Configuration


file1_out = f"atmospherictransparencygrid_rayleigh_training.npy"
file2_out = f"atmospherictransparencygrid_rayleigh_test.npy"


file3_out = f"atmospherictransparencygrid_O2abs_training.npy"
file4_out = f"atmospherictransparencygrid_O2abs_test.npy"


file5_out = f"atmospherictransparencygrid_PWVabs_training.npy"
file6_out = f"atmospherictransparencygrid_PWVabs_test.npy"


file7_out = f"atmospherictransparencygrid_OZabs_training.npy"
file8_out = f"atmospherictransparencygrid_OZabs_test.npy"


# ### wavelength


WLMIN=300.
WLMAX=1100.
WLBIN=1.
NWLBIN=int((WLMAX-WLMIN)/WLBIN)
WL=np.linspace(WLMIN,WLMAX,NWLBIN)


# ### airmass


AIRMASSMIN=1.0
AIRMASSMAX=2.6
#AIRMASSMAX=1.1
DAM = 0.1


airmasses = np.arange(AIRMASSMIN,AIRMASSMAX,DAM)


NAM=len(airmasses)


sequential_colors = sns.color_palette("hls", NAM)
sns.palplot(sequential_colors)


airmass_training = airmasses
airmass_test = airmasses + DAM/2.


NX=len(airmasses)
NY=NWLBIN


# ### PWV

PWVMIN = 0
PWVMAX = 11
DPWV = 0.5

pwv_training = np.arange(PWVMIN,PWVMAX,DPWV)
pwv_test = pwv_training + DPWV/2.

NPWV = len(pwv_training)


# ### OZON
OZMIN = 0
OZMAX = 600
DOZ   = 100

oz_training = np.arange(OZMIN,OZMAX,DOZ)
oz_test = oz_training  + DOZ/2.

NOZ = len(oz_training)


# ### Data

data_O2abs_training=np.zeros((NWLBIN,NAM))
data_O2abs_test=np.zeros((NWLBIN,NAM))

data_rayleigh_training=np.zeros((NWLBIN,NAM))
data_rayleigh_test=np.zeros((NWLBIN,NAM))


data_H2Oabs_training=np.zeros((NWLBIN,NAM,NPWV))
data_H2Oabs_test=np.zeros((NWLBIN,NAM,NPWV))

data_OZabs_training=np.zeros((NWLBIN,NAM,NOZ))
data_OZabs_test=np.zeros((NWLBIN,NAM,NOZ))


pwv= 0
pwv= 0
oz = 0

for idx,am in enumerate(airmass_training):
    path,thefile = libsimulateVisible.ProcessSimulation(am,pwv,oz,0,prof_str='us',proc_str='sc',cloudext=0.0, FLAG_VERBOSE=False)
    data = np.loadtxt(os.path.join(path,thefile))
    f = interpolate.interp1d(x=data[:,0], y=data[:,1],fill_value="extrapolate")
    atm=f(WL)
    data_rayleigh_training[:,idx]=atm
    

np.save(file1_out,data_rayleigh_training, allow_pickle=False)


#plt.plot(WL,data_rayleigh_training[:,:])

for idx,am in enumerate(airmass_test):
    path,thefile = libsimulateVisible.ProcessSimulation(am,pwv,oz,0,prof_str='us',proc_str='sc',cloudext=0.0, FLAG_VERBOSE=False)
    data = np.loadtxt(os.path.join(path,thefile))
    f = interpolate.interp1d(x=data[:,0], y=data[:,1],fill_value="extrapolate")
    atm=f(WL)
    data_rayleigh_test[:,idx]=atm


np.save(file2_out,data_rayleigh_test, allow_pickle=False)

#plt.plot(WL,data_rayleigh_test[:,:])


for idx,am in enumerate(airmass_training):
    path,thefile = libsimulateVisible.ProcessSimulation(am,pwv,oz,0,prof_str='us',proc_str='ab',cloudext=0.0, FLAG_VERBOSE=False)
    data = np.loadtxt(os.path.join(path,thefile))
    f = interpolate.interp1d(x=data[:,0], y=data[:,1],fill_value="extrapolate")
    atm=f(WL)
    data_O2abs_training[:,idx]=atm


np.save(file3_out,data_O2abs_training, allow_pickle=False)

# plt.plot(WL,data_O2abs_training[:,:])


for idx,am in enumerate(airmass_test):
    path,thefile = libsimulateVisible.ProcessSimulation(am,pwv,oz,0,prof_str='us',proc_str='ab',cloudext=0.0, FLAG_VERBOSE=False)
    data = np.loadtxt(os.path.join(path,thefile))
    f = interpolate.interp1d(x=data[:,0], y=data[:,1],fill_value="extrapolate")
    atm=f(WL)
    data_O2abs_test[:,idx]=atm


np.save(file4_out,data_O2abs_test, allow_pickle=False)

#plt.plot(WL,data_O2abs_test[:,:])


# ## Precipitable water vapor


oz=0
for idx_pwv,pwv in enumerate(pwv_training):
    data_slice_training=np.zeros((NWLBIN,NAM))
    for idx_am,am in enumerate(airmass_training):     
        path,thefile = libsimulateVisible.ProcessSimulation(am,pwv,oz,0,prof_str='us',proc_str='ab',cloudext=0.0, FLAG_VERBOSE=False)
        data = np.loadtxt(os.path.join(path,thefile))
        f = interpolate.interp1d(x=data[:,0], y=data[:,1],fill_value="extrapolate")
        atm=f(WL)
        data_slice_training[:,idx_am]=atm
        
    data_slice_training/=data_O2abs_training
    data_H2Oabs_training[:,:,idx_pwv] = data_slice_training
       
np.save(file5_out,data_H2Oabs_training, allow_pickle=False)

oz=0
for idx_pwv,pwv in enumerate(pwv_test):
    data_slice_test=np.zeros((NWLBIN,NAM))
    for idx_am,am in enumerate(airmass_test):     
        path,thefile = libsimulateVisible.ProcessSimulation(am,pwv,oz,0,prof_str='us',proc_str='ab',cloudext=0.0, FLAG_VERBOSE=False)
        data = np.loadtxt(os.path.join(path,thefile))
        f = interpolate.interp1d(x=data[:,0], y=data[:,1],fill_value="extrapolate")
        atm=f(WL)
        data_slice_test[:,idx_am]=atm
        
    data_slice_test/=data_O2abs_test
    data_H2Oabs_test[:,:,idx_pwv] = data_slice_test


np.save(file6_out,data_H2Oabs_test,allow_pickle=False)


# ## Ozone


pwv=0
for idx_oz,oz in enumerate(oz_training):
    data_slice_training=np.zeros((NWLBIN,NAM))
    for idx_am,am in enumerate(airmass_training):     
        path,thefile = libsimulateVisible.ProcessSimulation(am,pwv,oz,0,prof_str='us',proc_str='ab',cloudext=0.0, FLAG_VERBOSE=False)
        data = np.loadtxt(os.path.join(path,thefile))
        f = interpolate.interp1d(x=data[:,0], y=data[:,1],fill_value="extrapolate")
        atm=f(WL)
        data_slice_training[:,idx_am]=atm
        
    data_slice_training/=data_O2abs_training
    data_OZabs_training[:,:,idx_oz] = data_slice_training


np.save(file7_out,data_OZabs_training, allow_pickle=False)


pwv=0
for idx_oz,oz in enumerate(oz_test):
    data_slice_test=np.zeros((NWLBIN,NAM))
    for idx_am,am in enumerate(airmass_test):     
        path,thefile = libsimulateVisible.ProcessSimulation(am,pwv,oz,0,prof_str='us',proc_str='ab',cloudext=0.0, FLAG_VERBOSE=False)
        data = np.loadtxt(os.path.join(path,thefile))
        f = interpolate.interp1d(x=data[:,0], y=data[:,1],fill_value="extrapolate")
        atm=f(WL)
        data_slice_test[:,idx_am]=atm
        
    data_slice_test/=data_O2abs_test
    data_OZabs_test[:,:,idx_oz] = data_slice_test


np.save(file8_out,data_OZabs_test, allow_pickle=False)

