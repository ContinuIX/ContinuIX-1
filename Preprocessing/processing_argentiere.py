#%% Check and pre-processing of data
''' Preprocessing of observational data for the Argentiere glacier.
Observational data has been submitted by data contributor XXX.

The data has been checked and cleaned.
The data is stored (a) as a "RAW" version and (b) as a "HOMOGENIZED" version. 
The RAW version is the data as submitted by the data contributor, with only minor cleaning (e.g. removing NaN values, reprojecting to a common CRS, unit value, etc.). The files are saved in either .tif or .shp format, depending on the data type.
The HOMOGENIZED version is the data that has been homogenized to a common grid, resolution and CRS. The homogenized data is saved to a netcdf file (containing all parameters) for further use in the ContinuIX project. 

Note that the concept of cleaning and homogenizing is the same for all glaciers, but depending on the provided data the undertaken steps may differ.

# M. Izeboud, June 2026
'''

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import os
import matplotlib.pyplot as plt
import rasterio as rio
import rioxarray #  activates .rio accessor of xarray

path2data_raw = '../../ContinuIX_WP1_data/Data_Package/01_submitted_data/Argentiere/'
path2data_clean = '../../ContinuIX_WP1_data/Data_Package/02_raw-cleaned_data/Argentiere/'
path2data_homog = '../../ContinuIX_WP1_data/Data_Package/03_homogenized_data/Argentiere/'

import datafunctions as datafuncs

#%% Step 0: Check submitted (raw) data
''' ##################################
Check submitted (raw) data where needed

## DHDT: 2012-2021 in m/yr. No cleaning needed.

## Velocity: no cleaning needed.

## Thickness: 
profiles.csv is from 2017 and is used to obtain the interpolated H.tiffs (from readme)
But which thickness grid to choose?
- argentiere_h_interpIGM_20170215: is interpolated with thickness inversion 
    from IGM that also uses velocity (same as submitted velocity data)
    --> do not use, circularity of approach?
- argentiere_h_interpSGS_Farinotti2019_20170215: is interpolated using 
    Sequential Gaussian Simulations (SGS) and the Farinotti et al. (2019) bed
    --> can use (?) although Farinotti2019 bed is also obtained from models with thickness inversion..
- argentiere_h_interpSGS_SIA_20170215: is interpolated using SGS
     and a bed estimated using the SIA from the Pléiades surface velocities
     --> do not use, circularity of approach?
--> use interpolation using GlaTE

## Thickness profiles: 
csv in EPSG4326, convert to shapefile and to EPSG32632


## Bedrock: not provided
Calculate as `argentiere_bedrock = argentiere_DEM_20170215 - argentiere_h_20170215_SGS-Farinotti`.
Before calculating difference, the thickenss grid is upsampled to the DEM grid 


---
TO DO: check grid resolutions and homogenize these

##################################
'''
target_crs = 'EPSG:32632'

''' ##################################
Load outlines
################################## '''
outline_2012 = gpd.read_file(os.path.join(path2data_raw, 'argentiere_outline_20120819.shp'))
outline_2020 = gpd.read_file(os.path.join(path2data_raw, 'argentiere_outline_20200908.shp'))
assert outline_2012.crs == outline_2020.crs == target_crs, "CRS of outlines do not match"

## get extent of outlines
extent_2012 = outline_2012.total_bounds # returns (minx, miny, maxx, maxy)
extent_2020 = outline_2020.total_bounds

#%%
''' ##################################
Thickness profiles
################################## '''

df_profiles = pd.read_csv(os.path.join(path2data_raw, 'argentiere_h_20170215.csv'),
                        #   skiprows=1, # skip first row with metadata
                          header=0, # use first row as header
                          delimiter=';')
df_profiles.head()
df_profiles['lat'] = df_profiles['latitude'].astype(float)
df_profiles['lon'] = df_profiles['longitude'].astype(float)

## convert to geodataframe, set geometry and CRS
crs_profiles = 'EPSG:4326'
gdf_profiles4326 = gpd.GeoDataFrame(df_profiles,
                                geometry=gpd.points_from_xy(df_profiles['lon'], df_profiles['lat']),
                                crs=crs_profiles)
## reproject to EPSG32632
gdf_profiles = gdf_profiles4326.to_crs('EPSG:32632')

## to save to SHP, column names longer than 10char will be truncated.
# identify columns longer than 10 char 
column_names = gdf_profiles.columns
long_columns = [col for col in column_names if len(col) > 10]
print(f"Columns longer than 10 characters: {long_columns}") 
# ['elevation_date', 'max_elevation_date', 'thickness_uncertainty']

# rename columns to be shorter than 10 char
gdf_profiles = gdf_profiles.rename(columns={
    'elevation_date': 'elev_date',
    # 'max_elevation_date': 'max_elevDate', ## drop this column since there's only one value (15/02/2017)
    'thickness_uncertainty': 'h_uncert',
}).drop(columns=['max_elevation_date']) # drop this column since there's only one value (15/02/2017)

## save as shapefile in cleaned data directory
fname_profiles = 'argentiere_h_profiles_20170215.shp'
if not os.path.exists(os.path.join(path2data_clean, fname_profiles)):
    gdf_profiles.to_file(os.path.join(path2data_clean, fname_profiles))
else:
    print(f"File {fname_profiles} already exists in cleaned data directory. Skipping save.")

#%%
''' ##################################
Bedrock, DEM, thickness
################################## '''

da_dem17 = xr.open_dataarray(os.path.join(path2data_raw, 'argentiere_DEM_20170215.tif')
                             ).isel(band=0).drop_vars('band')
da_h17 = xr.open_dataarray(os.path.join(path2data_clean, 'argentiere_h_2017_glate.tif')
                           ).isel(band=0).drop_vars('band')

assert da_dem17.rio.crs == da_h17.rio.crs, "CRS of DEM and thickness grid do not match"
assert da_dem17.rio.resolution() == da_h17.rio.resolution(), "Resolution of DEM and thickness grid do not match"
print(f"CRS of DEM and thickness grid: {da_dem17.rio.crs}")

## DEM17 is larger area outside of argentiere, clip to argentiere extent using 2012 outline (alrgest outlines)
da_argentiere_4m = da_dem17.rio.clip(outline_2012.geometry, outline_2012.crs, drop=True, all_touched=True)
## make it a nicer regular grid
da_argentiere_4m = datafuncs.create_regular_dummy_grid(da_argentiere_4m, grid_res=4, crs=target_crs, unit='m')


## match DEM and H grid extents to this one
da_h17   = datafuncs.reproject_match_grid(da_argentiere_4m, da_h17, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)
da_dem17 = datafuncs.reproject_match_grid(da_argentiere_4m,da_dem17, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)

assert da_dem17.rio.resolution() == da_h17.rio.resolution(), "Resolution of DEM and thickness grid do not match"
assert da_dem17.shape == da_h17.shape, "Shape of DEM and thickness grid do not match"

## BEDROCK: calculate difference, fill areas outside of glacier (where H is nan or 0) with DEM values (assuming bedrock = DEM there)
da_bedrock = da_dem17 - da_h17
da_bedrock_filled = xr.where(np.isnan(da_bedrock), 
                               da_dem17,  ## whre condition is ture
                               da_bedrock) ## where condition is false

## save to CLEAN directory
da_bedrock_filled.attrs = {
    'description': 'argentiere bedrock DEM calculated as argentiere_DEM_20170215 - argentiere_h_2017_glate.tif',
    'units': 'm',
    'crs': str(da_bedrock_filled.rio.crs),
    'resolution': str(da_bedrock_filled.rio.resolution()),
}

fname = 'argentiere_bedrock.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    da_bedrock_filled.rio.to_raster(os.path.join(path2data_clean, fname))

da_dem17.attrs = {
    'description': 'Surface elevation',
    'units': 'm',
    'crs': str(da_dem17.rio.crs),
    'resolution': str(da_dem17.rio.resolution()),
}
fname = 'argentiere_DEM_20170215.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    da_dem17.rio.to_raster(os.path.join(path2data_clean, fname))

## tmp: re-save thickenss grid to CLEAN directory
da_h17.attrs = {
    'description': 'Ice thickness interpolated from GPR data (GlaTE)',
    'units': 'm',
    'crs': str(da_h17.rio.crs),
    'resolution': str(da_h17.rio.resolution()),
}
fname = 'argentiere_h_2017_glate.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    da_h17.rio.to_raster(os.path.join(path2data_clean, fname))

# %%
''' ##################################
dhdt, vx, vy:
no cleaning needed, but need to clip to argentiere extent.
################################## '''

da_dhdt_in = xr.open_dataarray(os.path.join(path2data_raw, 'argentiere_dhdt_2012-2021.tif')
                               ).isel(band=0).drop_vars('band')

## Match extent as DEM and H grid
assert da_dhdt_in.rio.crs == da_dem17.rio.crs == target_crs, "CRS of dhdt and DEM do not match"

## dhdt: 4m resolution
da_dhdt_clean = datafuncs.reproject_match_grid(da_argentiere_4m, da_dhdt_in, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)

## Assign attributes
da_dhdt_clean.attrs = {'long_name':'Surface Elevation Change',
                              'units':'m/year',
                              'crs': str(da_dhdt_clean.rio.crs),
                              'timestamp':'2012-2021',
                              'description':'annual elevation change',
                              }

## save to CLEAN directory
fname = 'argentiere_dhdt_2012-2021.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    da_dhdt_clean.rio.to_raster(os.path.join(path2data_clean, fname))

## get extent of grid
print('argentiere extent 4m ', da_dhdt_clean.rio.bounds()) # returns (minx, miny, maxx, maxy)

'''## velocity: 20m resoltuion'''
da_vx_in = xr.open_dataarray(os.path.join(path2data_raw, 'argentiere_velx_2012-2021.tif')
                               ).isel(band=0).drop_vars('band')
da_vy_in = xr.open_dataarray(os.path.join(path2data_raw, 'argentiere_vely_2012-2021.tif')
                               ).isel(band=0).drop_vars('band')

da_argentiere_20m = datafuncs.create_regular_dummy_grid(da_argentiere_4m, grid_res=20, crs=target_crs, unit='m')
da_vx_clean = datafuncs.reproject_match_grid(da_argentiere_20m, da_vx_in, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
da_vy_clean = datafuncs.reproject_match_grid(da_argentiere_20m, da_vy_in, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)

da_vx_clean.attrs = {'long_name':'Surface ice velocity',
                              'units':'m/year',
                              'crs': str(da_vx_clean.rio.crs),
                              'timestamp':'2012-2021',
                              'description':'Surface ice flow velocity (x-component)',
                              }
da_vy_clean.attrs = {'long_name':'Surface ice velocity',
                              'units':'m/year',
                              'crs': str(da_vy_clean.rio.crs),
                              'timestamp':'2012-2021',
                              'description':'Surface ice flow velocity (y-component)',
                              }
## save to CLEAN directory
fname_vx = 'argentiere_vx.tif'
fname_vy = 'argentiere_vy.tif'
if not os.path.exists(os.path.join(path2data_clean, fname_vx)):
    da_vx_clean.rio.to_raster(os.path.join(path2data_clean, fname_vx))
    da_vy_clean.rio.to_raster(os.path.join(path2data_clean, fname_vy))

#%%

''' ##################################
Make Elevation bins
--> 50 m binstep
--> based on earliest DEM if multiple available (assuming glacier is retreating, so earliest DEM has highest elevations)
--> do for both CLEAN and HOMOGNEIZED data; so possibly also resampling to different resolution
##################################
'''

## multiple DEMs: use first to do the binning, but use the min-max range of both to define bin range
# hmin = np.min([da_dem17.min().item(), da_dem23.min().item()]) 
# hmax = np.max([da_dem17.max().item(), da_dem23.max().item()]) 
hmin = da_dem17.min().item()
hmax = da_dem17.max().item()
da_elev_bins, elev_bin_edges = datafuncs.dicretize_elevation_bins(da_dem17, 
                                                     hmin=hmin, hmax=hmax,
                                                     binstep=50)

## for HOMOGENIZED: do not downsample elev-bin dataArray, but do new binning on donwsampled DEM
da_dem17_20m = datafuncs.reproject_match_grid(da_argentiere_20m, da_dem17, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
da_elev_bins_20m, elev_bin_edges_20m = datafuncs.dicretize_elevation_bins(da_dem17_20m,
                                                     hmin=hmin, hmax=hmax,
                                                     binstep=50)


## save to CLEAN directory
fname = 'argentiere_elev-bins.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    da_elev_bins.rename('elevation_bins').rio.to_raster(os.path.join(path2data_clean, fname))


#%%
''' ##################################
Global uncertainties
Thickness: from H profiles, using thk_min and thk_max to infer [thk] uncertainty range. This is provided over all points --> take average as global value
################################## '''

'''# unct_thickness : provided in readme '''
unct_thickness = 20 # m
'''unct_dhdt --> provided in readme '''
unct_dhdt = 0.07 # m/yr   
'''unct_velo --> provided in readme''' 
unct_velo = 3 # m yr−1
unct_dem = 0.1 # m

#%%
''' ##################################
HOMOGENIZED DATA
- fill all NaN values with 0
- do something else for DEM
################################## '''

target_res = 20 # meter
da_dummy_target = da_argentiere_20m.copy()


'''## OUTLINE TO MASK'''
# burn outline into raster mask
gdf_outline1 = outline_2012
gdf_outline2 = outline_2020

da_outline_mask1 = (da_dummy_target*2012).rio.clip(gdf_outline1.geometry, gdf_outline1.crs, drop=False) # drop=False to keep the same grid and not drop the pixels outside the outline (which will be set to nodata)
da_outline_mask2 = (da_dummy_target*2020).rio.clip(gdf_outline2.geometry, gdf_outline2.crs, drop=False) # drop=False to keep the same grid and not drop the pixels outside the outline (which will be set to nodata)
## combine to single dataArray
da_outline_mask = xr.concat([da_outline_mask1, da_outline_mask2], 
                            dim='time').max(dim='time') 

''' ##################################
ASSEMBLE NETCDF
- DEM
- thickness
- vx
- vy
- dhdt
- bedrock
- outline: as mask
Fill NaN values with another nodata value 
--> can be 0 for all variables except DEM and bedrock, so need to make sure these don't have missing values
##################################
'''


## select variables to use for saving
# da_dem_hmg = da_dem17_20m.copy()
# da_bedrock_hmg = da_bedrock_filled.copy()
# da_thickness_hmg = da_h17.copy()
# da_dhdt_hmg = da_dhdt_clean.copy()
# da_vx_hmg = da_vx_clean.copy()
# da_vy_hmg = da_vy_clean.copy()
# da_elev_bins_hmg = da_elev_bins_20m.copy()

## initial check that all variables have the same CRS, resolution and shape
da_var_dict = {'BED':da_bedrock_filled.copy().rename('BED'),
                'DEM': da_dem17_20m.copy().rename('DEM'),
                'ELEVBINS': da_elev_bins_20m.copy().rename('ELEVBINS'),
                'THK':da_h17.copy().rename('THK'),
                'DHDT': da_dhdt_clean.copy().rename('DHDT'),
                'VX': da_vx_clean.copy().rename('VX'),
                'VY': da_vy_clean.copy().rename('VY'),
                'ICEMASK': da_outline_mask.copy().rename('ICEMASK'),
                # 'UNCT_VX': ##,
                # 'UNCT_VY': ##,
}
## resample to target grid where necessary
for varname , var in da_var_dict.items():
    if var.rio.resolution()[0] != target_res:
        print(f'.. resampling {varname} from {var.rio.resolution()[0]} m to {target_res} m')
        var_target_res = datafuncs.reproject_match_grid(da_dummy_target, var, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)

        ## put back in dictionary
        da_var_dict[varname] = var_target_res
    print(var_target_res.rio.resolution(), var_target_res.shape)
    if var_target_res.shape != da_dummy_target.shape:
        raise ValueError(f"Shape of {varname} does not match target shape: {var_target_res.shape} vs {da_dummy_target.shape}")
    
assert all(da.rio.crs == da_dummy_target.rio.crs for da in da_var_dict.values()), "Not all variables have the same CRS"
assert all(da.rio.resolution() == da_dummy_target.rio.resolution() for da in da_var_dict.values()), "Not all variables have the same resolution"
assert all(da.shape == da_dummy_target.shape for da in da_var_dict.values()), "Not all variables have the same shape"


'''
# SET ATTRIBUTES OF VARIABLES
Handle NaN values 
'''

da_outline_mask = (da_var_dict['ICEMASK'].copy()
                #    .fillna(0) # fill NaN values with 0 (outside outline)
                #    .rename('ICEMASK')
                   .assign_attrs({'long_name':'Glacier Outline Mask',
                                  'units':'year',
                                   'uncertainty': 'unknown', 
                                  'crs':target_crs,
                                  'timestamp':'2012 and 2020',
                                  'description': 'Value is max year of valid glaciated pixel; 0 for non-glaciated pixels.',
                                  'nodata': 0})
                    .rio.write_crs(target_crs)
)


# DEM and Bedrock: should not have NaN values 
# if da_dem_hmg.isnull().any():
#     # raise ValueError("DEM has NaN values.")
#     da_dem_hmg = da_dem_hmg.fillna(-999) # fill NaN values with -999 
# else: 
da_dem_hmg = (da_var_dict['DEM'].copy()
                # .fillna(-999)
                # .rename('DEM') 
                .assign_attrs({'long_name':'Elevation',
                                'units':'m',
                                'uncertainty': f'+- {unct_dem} m',
                                'crs':target_crs,
                                'timestamp':'2017',
                                'description':'Elevation data.'
                                })
                    # .drop_vars('spatial_ref')
                    .rio.write_crs(target_crs)
    )

# if da_bedrock_hmg.isnull().any():
#     # raise ValueError("Bedrock has NaN values, cannot assign nodata value of 0.")
#     # da_bedrock_hmg = da_bedrock_hmg.fillna(-999) # fill NaN values with 0
# else: 
da_bedrock_hmg = (da_var_dict['BED'].copy()
                    # .fillna(-999) # fill NaN values with -999
                    # .rename('BED')
                    .assign_attrs({'long_name':'Bedrock Elevation',
                                   'units':'m',
                                   'uncertainty': 'unknown', 
                                   'crs':target_crs,
                                   'timestamp':'2017',
                                   'description':'bedrock calculated from DEM and H. Where H=0, bedrock=DEM.'
                                   })
                    # .drop_vars('spatial_ref')
                    .rio.write_crs(target_crs)
    )

# if da_elev_bins_hmg.isnull().any():
#     raise ValueError("Elevation bins has NaN values (since DEM has them), cannot assign nodata value of 0.")
# else: 
da_elev_bins_hmg = (da_var_dict['ELEVBINS'].copy()
                    # .fillna(-999) # fill NaN values with -999
                    # .rename('ELEVBINS')
                    .assign_attrs({'long_name':'Elevation Bins',
                                  'units':'m',
                                   'uncertainty': 'unknown', 
                                  'crs':target_crs,
                                  'timestamp':'2017',
                                  'description': f'Discretized elevation values into bins of 50 m. Using lowest (left-edge) value for each bin.'
                                  })
                    .rio.write_crs(target_crs)
)

## thickness, dhdt, velo: can fill NaN with 0
da_thickness_hmg = (da_var_dict['THK'].copy()
                    .fillna(0)
                    # .rename('THK')
                    .assign_attrs({'long_name':'Ice Thickness',
                                   'units':'m',
                                   'uncertainty': f'+- {unct_thickness:.1f} m',
                                   'crs':target_crs,
                                   'timestamp':'20170215',
                                   'description':'ice thickness interpolated from GPR. Missing/NaN values were filled with 0.',
                                   'nodata': 0})
                .rio.write_crs(target_crs)
                    )
da_dhdt_hmg = (da_var_dict['DHDT'].copy()
               .fillna(0)
               # .rename('DHDT')
               .assign_attrs({'long_name':'Surface Elevation Change',
                              'units':'m/year',
                              'uncertainty': f'+- {unct_dhdt:.2f} m/year',
                              'crs':target_crs,
                              'timestamp':'2012-2021',
                              'description':'Annual elevation change. Missing/NaN values were filled with 0.',
                              'nodata': 0})
                .rio.write_crs(target_crs)
               )

da_vx_hmg = (da_var_dict['VX'].copy()
             .fillna(0)
             # .rename('VX')
             .assign_attrs({'long_name': 'surface ice velocity (x-component)',
                            'units':'m/year',
                              'uncertainty': f'+- {unct_velo:.1f} m/year',
                            'crs':target_crs,
                            'timestamp':'2012-2021',
                            'description':'Velocity for the period 2012-2021. Missing/NaN values were filled with 0.',
                            'nodata': 0
                            })
                .rio.write_crs(target_crs)
)

da_vy_hmg = (da_var_dict['VY'].copy()
             .fillna(0)
             # .rename('VY')
             .assign_attrs({'long_name': 'surface ice velocity (y-component)',
                            'units':'m/year',
                              'uncertainty': f'+- {unct_velo:.1f} m/year',
                            'crs':target_crs,
                            'timestamp':'2012-2021',
                            'description':'Velocity for 2012-2021. Missing/NaN values were filled with 0.',
                            'nodata': 0
                            })
                .rio.write_crs(target_crs)
)

#%%

## final check that everything is still homogenized 

da_var_list = [ da_bedrock_hmg,
                da_dem_hmg, 
                da_elev_bins_hmg,
                da_thickness_hmg,
                da_dhdt_hmg,
                da_vx_hmg,
                da_vy_hmg,
                da_outline_mask,
                ]
assert all(da.rio.crs == da_dummy_target.rio.crs for da in da_var_list), "Not all variables have the same CRS"
assert all(da.rio.resolution() == da_dummy_target.rio.resolution() for da in da_var_list), "Not all variables have the same resolution"
assert all(da.shape == da_dummy_target.shape for da in da_var_list), "Not all variables have the same shape"

ds_glacier_hmg = (xr.combine_by_coords(da_var_list, 
                                       compat='no_conflicts')
                    .assign_attrs({'title':'homogenized glacier observation data',
                               'grid_resolution':str(da_dummy_target.rio.resolution()),
                               'description':'see attributes of each variable',
                               'timestamp':'',
                            #    'nodata': 0,
                    })
                    .rio.set_spatial_dims(x_dim="x", y_dim="y") # Make sure spatial dims are known
                    # Write CRS and CF grid mapping to the whole dataset
                    .rio.write_crs(target_crs)
                    .rio.write_grid_mapping("spatial_ref")
                    .rio.write_transform()
)

# Force each real data variable to point to spatial_ref
for var in ds_glacier_hmg.data_vars:
    if var != "spatial_ref":
        ds_glacier_hmg[var].attrs["grid_mapping"] = "spatial_ref"

'''# check values by plotting
'''
# for var in ds_glacier_hmg.data_vars:
#     da_plot = ds_glacier_hmg[var]
#     # print(da_plot)
#     fig,ax=plt.subplots(figsize=(6,5))
    
#     da_plot.plot.imshow(ax=ax)
# ds_glacier_hmg
# Set the no-data value in the encoding dictionary

#%%
'''## save to netcdf'''

# ## clean fill/missing values before encoding --> no just keep original NaN coding.
# for var in ds_glacier_hmg.data_vars:
#     ds_glacier_hmg[var].attrs.pop("_FillValue", None)
#     ds_glacier_hmg[var].attrs.pop("missing_value", None)
#     ds_glacier_hmg[var].encoding.pop("_FillValue", None)
#     ds_glacier_hmg[var].encoding.pop("missing_value", None)

## encoding settings for compression and data type; same for all variables
comp = {"zlib": True, 
        "complevel": 5,  ## level of compression; higher number = more compression but slower read/write
        "dtype": "float32", ## 7 digits of precision 
        # "_FillValue": np.float32(-999), ## for remaining NaN values
        }
encoding = {var: comp for var in ds_glacier_hmg.data_vars if var != "spatial_ref"}  # Exclude spatial_ref from encoding
encoding["spatial_ref"] = {}  # No compression for spatial_ref

fname_nc = 'argentiere_glacier_observations.nc'

try:
    print('--> saving homogenized data to netcdf; overwriting if file exists')
    ds_glacier_hmg.to_netcdf(os.path.join(path2data_homog, fname_nc), 
                            mode='w', format='NETCDF4', 
                            engine='netcdf4',
                            encoding=encoding ## don't use encoding; although it compresses data size, it loses CRS info 
    )
    ds_glacier_hmg.close()

except PermissionError:
    print('--> CHECK INPUT WINDOW')
    answ = input(f"PermissionError to write {fname_nc}. Input Y to overwrite")
    if answ == 'Y' or answ == 'y':
        print('..removing existing and re-saving file')
        os.remove(os.path.join(path2data_homog, fname_nc))
        ds_glacier_hmg.to_netcdf(os.path.join(path2data_homog, fname_nc), 
                            mode='w', format='NETCDF4', 
                            engine='netcdf4',
                            encoding=encoding ## don't use encoding; although it compresses data size, it loses CRS info 
        )
        ds_glacier_hmg.close()
    else: print('..aborted saving file')


#%% check values by loading saved data & plotting


fname_nc = 'argentiere_glacier_observations.nc'

with xr.open_dataset(
        os.path.join(path2data_homog, fname_nc),
        decode_coords="all" # decode_coords="all" is important when reopening NetCDFs with rioxarray-style CRS metadata; otherwise the CRS may appear to be missing.
    ) as ds_glacier_loaded:
    
    print('CRS:', ds_glacier_loaded.rio.crs)
    print('spatial_ref attrs:', ds_glacier_loaded["spatial_ref"].attrs)
    assert ds_glacier_loaded.rio.crs is not None, "CRS is missing in the loaded dataset"

## check values by plotting
fig,axs=plt.subplots(2,4, figsize=(16,8))
row,col = 0,0
for var, cmap, vminmax in zip(  ['BED',     'DEM',     'ELEVBINS', 'THK', 'DHDT', 'VX', 'VY',   'ICEMASK'],
                                ['cividis','cividis','cividis',  'Blues', 'RdBu','PiYG','PiYG',  'viridis'],
                                [(1.8e3, 4e3), (1.8e3, 4e3), (1.8e3, 4e3), (0,500),  (-5,5), (-100,100),(-100,100),None]):
    if vminmax is not None:
        vmin, vmax = vminmax
    else:
        vmin, vmax = None, None

    da_plot = ds_glacier_loaded[var]
    # print(da_plot)
    # fig,ax=plt.subplots(figsize=(6,5))
    ax=axs[row,col]
    da_plot.plot.imshow(ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, cbar_kwargs={'shrink': 0.7})
    ax.set_title(var)
    col+=1
    if col >= 4:
        col = 0
        row += 1
[ax.set_aspect('equal') for ax in axs.flatten()];
[ax.set_axis_off() for ax in axs.flatten()];

fig.savefig(os.path.join(path2data_homog, 'argentiere_netcdf_vars.png'), dpi=300)
# %%
