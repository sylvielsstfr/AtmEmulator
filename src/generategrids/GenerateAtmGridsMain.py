#!/usr/bin/env python
# coding: utf-8

# # Generate grids


# Import some generally useful packages

import os
import numpy as np
from itertools import cycle, islice
import copy
import pickle

from scipy import interpolate
from libradtranpy import  libsimulateVisible
import sys,getopt


import warnings
warnings.filterwarnings('ignore')


# preselected observation sites 
Dict_Of_sitesAltitudes = {'LSST':2.663, # Rubin-LSST
                          'CTIO':2.207, # Cerro Tololo Inter-American Observatory
                          'OHP':0.65, # Observatoire de Haute Provence
                          'PDM':2.8905, # Observatoire du Pic du Midi
                          'OMK':4.205,  # Mauna Kea
                          'OSL':0,      # Sea Level
                           }

def usage():
    print("*******************************************************************")
    print(sys.argv[0],' -s<observation site-string> [-q <procs>]')
    print("Observation sites are : ")
    print(' '.join(Dict_Of_sitesAltitudes.keys()))
    print("procs =  sa, sc or ab")


    print('\t Actually provided : ')
    print('\t \t Number of arguments:', len(sys.argv), 'arguments.')
    print('\t \t Argument List:', str(sys.argv))


if __name__ == "__main__":


    try:
        opts, args = getopt.getopt(sys.argv[1:],"hs:q",["s=","q="])
    except getopt.GetoptError:
        print(' Exception bad getopt with :: '+sys.argv[0]+ ' -s<observation-site-string> -q<procs>')
        sys.exit(2)

    print('opts = ',opts)
    print('args = ',args)
        
        
    alt_str = ""
    proc_str = "sa"

    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-s", "--site"):
            alt_str = arg
        elif opt in ("-q"):
            proc_str = arg


        
    if alt_str in Dict_Of_sitesAltitudes.keys():
        OBS_tag = alt_str
    else:
        print(f"Observatory {alt_str} not in preselected observation site")
        print(f"This site {alt_str} must be added in libradtranpy preselected sites")
        sys.exit()

    FLAG_SCATTERING = True
    FLAG_ABSORPTION = True
    if proc_str == "sc":
        FLAG_ABSORPTION = False
    elif proc_str == "ab":
        FLAG_SCATTERING = False




    ########################
    ## Configuration
    ########################


    file01_out = f"{OBS_tag}_atmospherictransparencygrid_params_training.pickle"
    file02_out = f"{OBS_tag}_atmospherictransparencygrid_params_test.pickle"

    file1_out = f"{OBS_tag}_atmospherictransparencygrid_rayleigh_training.npy"
    file2_out = f"{OBS_tag}_atmospherictransparencygrid_rayleigh_test.npy"


    file3_out = f"{OBS_tag}_atmospherictransparencygrid_O2abs_training.npy"
    file4_out = f"{OBS_tag}_atmospherictransparencygrid_O2abs_test.npy"


    file5_out = f"{OBS_tag}_atmospherictransparencygrid_PWVabs_training.npy"
    file6_out = f"{OBS_tag}_atmospherictransparencygrid_PWVabs_test.npy"


    file7_out = f"{OBS_tag}_atmospherictransparencygrid_OZabs_training.npy"
    file8_out = f"{OBS_tag}_atmospherictransparencygrid_OZabs_test.npy"

    ####################
    # info dictionary

    info_params = {}


    info_params["OBS"] = OBS_tag

    ##################
    # ### wavelength
    ##################

    WLMIN=300.
    WLMAX=1100.
    WLBIN=1.
    NWLBIN=int((WLMAX-WLMIN)/WLBIN)
    WL=np.linspace(WLMIN,WLMAX,NWLBIN)

    info_params["WLMIN"] = WLMIN
    info_params["WLMAX"] = WLMAX
    info_params["WLBIN"] = WLBIN
    info_params["NWLBIN"] = NWLBIN
    info_params["WL"] = WL

    #################################
    # training and test dictionaries
    #################################

    info_params_training = copy.deepcopy(info_params)
    info_params_test = copy.deepcopy(info_params)

    ####################
    # ### airmass
    ####################

    AIRMASSMIN=1.0
    AIRMASSMAX=2.6
    DAM = 0.1

    airmasses = np.arange(AIRMASSMIN,AIRMASSMAX,DAM)
    NAM=len(airmasses)


    airmass_training = airmasses
    airmass_test = airmasses + DAM/2.


    NX=len(airmasses)
    NY=NWLBIN

    info_params_training["AIRMASSMIN"] = airmass_training.min()
    info_params_training["AIRMASSMAX"] = airmass_training.max()
    info_params_training["NAIRMASS"] = len(airmass_training)
    info_params_training["DAIRMASS"] = np.median(np.diff(airmass_training))
    info_params_training["AIRMASS"]  = airmass_training


    info_params_test["AIRMASSMIN"] = airmass_test.min()
    info_params_test["AIRMASSMAX"] = airmass_test.max()
    info_params_test["NAIRMASS"] = len(airmass_test)
    info_params_test["DAIRMASS"] = np.median(np.diff(airmass_test))
    info_params_test["AIRMASS"]  = airmass_test

    #########################
    # ### PWV
    #########################

    PWVMIN = 0
    PWVMAX = 11
    DPWV = 0.25

    pwv_training = np.arange(PWVMIN,PWVMAX,DPWV)
    pwv_test = pwv_training + DPWV/2.

    NPWV = len(pwv_training)

    info_params_training["PWVMIN"] = pwv_training.min()
    info_params_training["PWVMAX"] = pwv_training.max()
    info_params_training["NPWV"] = len(pwv_training)
    info_params_training["DPWV"] = np.median(np.diff(pwv_training))
    info_params_training["PWV"]  = pwv_training

    info_params_test["PWVMIN"] = pwv_test.min()
    info_params_test["PWVMAX"] = pwv_test.max()
    info_params_test["NPWV"] = len(pwv_test)
    info_params_test["DPWV"] = np.median(np.diff(pwv_test))
    info_params_test["PWV"]  = pwv_test

    # ### OZONE
    OZMIN = 0
    OZMAX = 600
    DOZ   = 25

    oz_training = np.arange(OZMIN,OZMAX,DOZ)
    oz_test = oz_training  + DOZ/2.

    NOZ = len(oz_training)

    info_params_training["OZMIN"] = oz_training.min()
    info_params_training["OZMAX"] = oz_training.max()
    info_params_training["NOZ"] = len(oz_training)
    info_params_training["DOZ"] = np.median(np.diff(oz_training))
    info_params_training["OZ"]  = oz_training

    info_params_test["OZMIN"] = oz_test.min()
    info_params_test["OZMAX"] = oz_test.max()
    info_params_test["NOZ"] = len(oz_test)
    info_params_test["DOZ"] = np.median(np.diff(oz_test))
    info_params_test["OZ"]  = oz_test

    with open(file01_out, 'wb') as handle:
        pickle.dump(info_params_training, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    with open(file02_out, 'wb') as handle:
        pickle.dump(info_params_test, handle, protocol=pickle.HIGHEST_PROTOCOL)   

    ##############################
    # ### Data
    ####################################

    data_O2abs_training=np.zeros((NWLBIN,NAM))
    data_O2abs_test=np.zeros((NWLBIN,NAM))

    data_rayleigh_training=np.zeros((NWLBIN,NAM))
    data_rayleigh_test=np.zeros((NWLBIN,NAM))


    data_H2Oabs_training=np.zeros((NWLBIN,NAM,NPWV))
    data_H2Oabs_test=np.zeros((NWLBIN,NAM,NPWV))

    data_OZabs_training=np.zeros((NWLBIN,NAM,NOZ))
    data_OZabs_test=np.zeros((NWLBIN,NAM,NOZ))

    ##########################################
    # Simulation of Rayleigh scattering
    ##############################################

    # note we call function ProcessSimulation with proc_str='sc' which mean scattering only

    # set pwv and ozone to zero (no need here)

    if FLAG_SCATTERING:
        pwv= 0
        oz = 0

        #training
        for idx,am in enumerate(airmass_training):
            path,thefile = libsimulateVisible.ProcessSimulation(am,pwv,oz,0,prof_str='us',proc_str='sc',cloudext=0.0, altitude_str = OBS_tag,FLAG_VERBOSE=False)
            data = np.loadtxt(os.path.join(path,thefile))
            f = interpolate.interp1d(x=data[:,0], y=data[:,1],fill_value="extrapolate")
            atm=f(WL)
            data_rayleigh_training[:,idx]=atm
    
        np.save(file1_out,data_rayleigh_training, allow_pickle=False)


        #test
        for idx,am in enumerate(airmass_test):
            path,thefile = libsimulateVisible.ProcessSimulation(am,pwv,oz,0,prof_str='us',proc_str='sc',cloudext=0.0,altitude_str = OBS_tag ,FLAG_VERBOSE=False)
            data = np.loadtxt(os.path.join(path,thefile))
            f = interpolate.interp1d(x=data[:,0], y=data[:,1],fill_value="extrapolate")
            atm=f(WL)
            data_rayleigh_test[:,idx]=atm

        np.save(file2_out,data_rayleigh_test, allow_pickle=False)


    if FLAG_ABSORPTION:
        ##########################################
        # Simulation of O2 absorption
        ##############################################

        # note we call function ProcessSimulation with proc_str='ab' which mean absorption only

        # set pwv and ozone to zero (no need here)
        pwv= 0
        oz = 0


        print("======================================")
        print("Simulation of O2 training sample")
        print("======================================")

        for idx,am in enumerate(airmass_training):
            path,thefile = libsimulateVisible.ProcessSimulation(am,pwv,oz,0,prof_str='us',proc_str='ab',cloudext=0.0,altitude_str = OBS_tag ,FLAG_VERBOSE=False)
            data = np.loadtxt(os.path.join(path,thefile))
            f = interpolate.interp1d(x=data[:,0], y=data[:,1],fill_value="extrapolate")
            atm=f(WL)
            data_O2abs_training[:,idx]=atm

        np.save(file3_out,data_O2abs_training, allow_pickle=False)
        print(f"...... O2 abs training file {file3_out} written")

        print("======================================")
        print("Simulation of O2 test sample")
        print("======================================")

        for idx,am in enumerate(airmass_test):
            path,thefile = libsimulateVisible.ProcessSimulation(am,pwv,oz,0,prof_str='us',proc_str='ab',cloudext=0.0, altitude_str = OBS_tag, FLAG_VERBOSE=False)
            data = np.loadtxt(os.path.join(path,thefile))
            f = interpolate.interp1d(x=data[:,0], y=data[:,1],fill_value="extrapolate")
            atm=f(WL)
            data_O2abs_test[:,idx]=atm

        np.save(file4_out,data_O2abs_test, allow_pickle=False)
        print(f"...... O2 abs test file {file4_out} written")

        ##########################################
        # Simulation of H2O absorption
        ##############################################

        # ## Precipitable water vapor
        print("======================================")
        print("Simulation of PWV training sample")
        print("======================================")

        # note we call function ProcessSimulation with proc_str='ab' which mean absorption only
        # we cut ozone
        # however we cannot cut O2 absorption

        oz=0
        for idx_pwv,pwv in enumerate(pwv_training):
            data_slice_training=np.zeros((NWLBIN,NAM))
            for idx_am,am in enumerate(airmass_training):     
                path,thefile = libsimulateVisible.ProcessSimulation(am,pwv,oz,0,prof_str='us',proc_str='ab',cloudext=0.0, altitude_str = OBS_tag ,FLAG_VERBOSE=False)
                data = np.loadtxt(os.path.join(path,thefile))
                f = interpolate.interp1d(x=data[:,0], y=data[:,1],fill_value="extrapolate")
                atm=f(WL)
                data_slice_training[:,idx_am]=atm
            # remove O2 absorption in H2O absorption profile
            data_slice_training/=data_O2abs_training
            data_H2Oabs_training[:,:,idx_pwv] = data_slice_training

        np.save(file5_out,data_H2Oabs_training,allow_pickle=False)
        print(f"...... H2O abs training file {file5_out} written")
       


        print("======================================")
        print("Simulation of PWV test sample")
        print("======================================")
        oz=0
        for idx_pwv,pwv in enumerate(pwv_test):
            data_slice_test=np.zeros((NWLBIN,NAM))
            for idx_am,am in enumerate(airmass_test):     
                path,thefile = libsimulateVisible.ProcessSimulation(am,pwv,oz,0,prof_str='us',proc_str='ab',cloudext=0.0, altitude_str = OBS_tag ,FLAG_VERBOSE=False)
                data = np.loadtxt(os.path.join(path,thefile))
                f = interpolate.interp1d(x=data[:,0], y=data[:,1],fill_value="extrapolate")
                atm=f(WL)
                data_slice_test[:,idx_am]=atm
            # remove O2 absorption in H2O absorption profile
            data_slice_test/=data_O2abs_test
            data_H2Oabs_test[:,:,idx_pwv] = data_slice_test


        np.save(file6_out,data_H2Oabs_test,allow_pickle=False)
        print(f"...... H2O abs test file {file6_out} written")



        ##########################################
        # Simulation of Ozone absorption
        ##############################################

        print("======================================")
        print("Simulation of Ozone training sample")
        print("======================================")

        # note we call function ProcessSimulation with proc_str='ab' which mean absorption only
        # we remove pwv absorption
        # however we cannot cut O2 absorption
        pwv=0
        for idx_oz,oz in enumerate(oz_training):
            data_slice_training=np.zeros((NWLBIN,NAM))
            for idx_am,am in enumerate(airmass_training):     
                path,thefile = libsimulateVisible.ProcessSimulation(am,pwv,oz,0,prof_str='us',proc_str='ab',cloudext=0.0, altitude_str = OBS_tag ,FLAG_VERBOSE=False)
                data = np.loadtxt(os.path.join(path,thefile))
                f = interpolate.interp1d(x=data[:,0], y=data[:,1],fill_value="extrapolate")
                atm=f(WL)
                data_slice_training[:,idx_am]=atm
            # remove O2 profile from ozone profile
            data_slice_training/=data_O2abs_training
            data_OZabs_training[:,:,idx_oz] = data_slice_training


        np.save(file7_out,data_OZabs_training, allow_pickle=False)
        print(f"...... O3 abs training file {file7_out} written")

        print("======================================")
        print("Simulation of Ozone test sample")
        print("======================================")
        pwv=0
        for idx_oz,oz in enumerate(oz_test):
            data_slice_test=np.zeros((NWLBIN,NAM))
            for idx_am,am in enumerate(airmass_test):     
                path,thefile = libsimulateVisible.ProcessSimulation(am,pwv,oz,0,prof_str='us',proc_str='ab',cloudext=0.0, altitude_str = OBS_tag ,FLAG_VERBOSE=False)
                data = np.loadtxt(os.path.join(path,thefile))
                f = interpolate.interp1d(x=data[:,0], y=data[:,1],fill_value="extrapolate")
                atm=f(WL)
                data_slice_test[:,idx_am]=atm
            # remove O2 profile from ozone profile
            data_slice_test/=data_O2abs_test
            data_OZabs_test[:,:,idx_oz] = data_slice_test


        np.save(file8_out,data_OZabs_test, allow_pickle=False)
        print(f"...... O3 abs test file {file8_out} written")

