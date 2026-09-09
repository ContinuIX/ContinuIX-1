#%%
''' Assemble all data to the respective EXP files
EXP01    : REF experiment, using homogenized data of all glaciers (real + synthetic) 
EXP02    : RAW experiment, using raw observation data that includes gaps and missing values (only real glaciers)
EXP03-15 : PERTUBATION experiments. Generated from the homogenized files --> see generate_pertubations.py
EXP16-20 : GLOBAL experiments, replacing observations with data from globally available datasets (only replacing thickness, dhdt, and velocity; only real glaciers).
'''

# M. Izeboud, July 2026

import numpy as np
import geopandas as gpd
import xarray as xr
import os
import matplotlib.pyplot as plt
import rasterio as rio
import rioxarray #  activates .rio accessor of xarray
import warnings

path2data_ref = '../../ContinuIX_WP1_data/Data_Package/03_homogenized_data/'
path2data_exp = '../../ContinuIX_WP1_data/EXP_Package/'

import datafunctions as datafuncs

#%% EXP 01
''' -------------------------
EXP 01 - REFERENCE
----------------------------- '''

