#%% Check and pre-processing of data
''' Preprocessing of observational data for the Sary-Tor glacier.
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

path2data_raw = '../../ContinuIX_WP1_data/Data_Package/01_submitted_data/SaryTor/'
path2data_clean = '../../ContinuIX_WP1_data/Data_Package/02_raw-cleaned_data/SaryTor/'
path2data_homog = '../../ContinuIX_WP1_data/Data_Package/03_homogenized_data/SaryTor/'

import datafunctions as datafuncs

#%% Step 0: Check submitted (raw) data
''' ##################################
Check submitted (raw) data where needed


## DEMs
- have DEMs annually, 2021-2025.
- resolutions vary between 0.073, 0.069, 0.075, 2 (2024), 0.081
--> resample all execpt 2024 to 0.08 m; keep 2024 as is for CLEANED dir. 
--> Update: resample all to 2m instead of 0.08 for CLEANED dir.
--> for homogenized dir, resample all to decided resolution (maybe 2m or maybe something else)

## DHDT: only DEMs provided, so need to calculate DHDT from DEMs.
- calculate dhdt for each year, but at 2m resolution (so to have the same resolution for every year)
- calculate average annual dhdt for 2021-2025


## Velocity: no cleaning needed.

## Thickness: no cleaning needed


## Thickness profiles: 

## Bedrock: not provided;
Thickenss is from 2021 so use DEM from 2021 to calculate bedrock.
Calculate as `sarytor_bedrock = sarytor_DEM_2021 - sarytor_h_2021`.
---
TO DO: check grid resolutions and homogenize these

##################################
'''
sarytor_outline = gpd.read_file(os.path.join(path2data_raw, 'sarytor_outline_2021.shp'))


''' ##################################
DEMs
################################## '''
da_dem21 = xr.open_mfdataset(os.path.join(path2data_raw, 'sarytor_DEM_2021.tif')
                             ).isel(band=0).drop_vars('band')['band_data']
da_dem22 = xr.open_mfdataset(os.path.join(path2data_raw, 'sarytor_DEM_2022.tif')
                             ).isel(band=0).drop_vars('band')['band_data']
da_dem23 = xr.open_mfdataset(os.path.join(path2data_raw, 'sarytor_DEM_2023.tif')
                             ).isel(band=0).drop_vars('band')['band_data']
da_dem24 = xr.open_mfdataset(os.path.join(path2data_raw, 'sarytor_DEM_2024.tif')
                             ).isel(band=0).drop_vars('band')['band_data']
da_dem25 = xr.open_mfdataset(os.path.join(path2data_raw, 'sarytor_DEM_2025.tif')
                             ).isel(band=0).drop_vars('band')['band_data']

assert da_dem21.rio.crs == da_dem22.rio.crs == da_dem23.rio.crs == da_dem24.rio.crs == da_dem25.rio.crs, "CRS of DEMs do not match"
# assert da_dem21.shape == da_dem22.shape == da_dem23.shape == da_dem24.shape, "Shape of DEMs do not match"
# assert da_dem21.rio.resolution() == da_dem22.rio.resolution() == da_dem23.rio.resolution() == da_dem24.rio.resolution(), "Resolution of DEMs do not match"
print('CRS of DEMs:', da_dem21.rio.crs)
print(da_dem21.shape, da_dem22.shape, da_dem23.shape, da_dem24.shape, da_dem25.shape)
print(da_dem21.rio.resolution(), da_dem22.rio.resolution(), da_dem23.rio.resolution(), da_dem24.rio.resolution(), da_dem25.rio.resolution()) ## 7.3 cm

## set up empty but regular grid
# da_dummy = create_regular_dummy_grid(da_dem21, grid_res=0.08, crs=da_dem21.rio.crs) ## 8 cm resolution
# da_dem21_08 = reproject_match_grid(da_dummy, da_dem21, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan) ## use nearest, since resampling is minimal
# da_dem22_08 = reproject_match_grid(da_dummy, da_dem22, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan) ## use nearest, since resampling is minimal
# da_dem23_08 = reproject_match_grid(da_dummy, da_dem23, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan) ## use nearest, since resampling is minimal
# ## skip 2024
# da_dem25_08 = reproject_match_grid(da_dummy, da_dem25, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan) ## use nearest, since resampling is minimal

# resampled and save DEMs to cleaned data directory
# da_dummy = datafuncs.create_regular_dummy_grid(da_dem21, grid_res=0.08, crs=da_dem21.rio.crs) ## 8 cm resolution
da_dummy = datafuncs.create_regular_dummy_grid(da_dem21, grid_res=2, crs=da_dem21.rio.crs) ## 2 m resolution

for year, da_dem in zip([2021, 2022, 2023, 2024, 2025], [da_dem21, da_dem22, da_dem23, da_dem24, da_dem25]):
    fname = f'sarytor_dem_{year}_2m.tif'
    # da_dummy = create_regular_dummy_grid(da_dem21, grid_res=0.08, crs=da_dem21.rio.crs) ## 8 cm resolution

    if not os.path.exists(os.path.join(path2data_clean, fname)):
        print(f'Reprojecting {year}')
        da_dem_08 = datafuncs.reproject_match_grid(da_dummy, da_dem, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan) ## use nearest, since resampling is minimal
        print('CRS of DEMs:', da_dem_08.rio.crs, 'resolution:', da_dem_08.rio.resolution())
        print('Resolution of DEMs:', da_dem_08.rio.resolution())

        print(f'Saving {fname} to cleaned data directory...')
        da_dem_08.rio.to_raster(os.path.join(path2data_clean, fname))
    else:
        print(f"File {fname} already exists in cleaned data directory. Skipping save.")

# fname = f'sarytor_dem_2024_2m.tif'
# if not os.path.exists(os.path.join(path2data_clean, fname)):
#     print(f'Saving {fname} to cleaned data directory...')
#     da_dem24.rio.to_raster(os.path.join(path2data_clean, fname))
# else:
#     print(f"File {fname} already exists in cleaned data directory. Skipping save.")

#%%
''' ##################################
dhdt
################################## '''
## also resample dem23 and dem25 to 2m for dhdt_ w.r.t 2024
da_dem21_2m = datafuncs.reproject_match_grid(da_dem24, da_dem21, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan) ## use nearest, since resampling is minimal
da_dem22_2m = datafuncs.reproject_match_grid(da_dem24, da_dem22, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan) ## use nearest, since resampling is minimal
da_dem23_2m = datafuncs.reproject_match_grid(da_dem24, da_dem23, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan) ## use nearest, since resampling is minimal
da_dem25_2m = datafuncs.reproject_match_grid(da_dem24, da_dem25, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan) ## use nearest, since resampling is minimal

# ## plot only the 2m versions, plotting 8cm version takes too long
# # da_dem23_2m.plot.imshow()
# # sarytor_outline.boundary.plot(ax=plt.gca(), color='red')
# da_dem24.plot.imshow()
# sarytor_outline.boundary.plot(ax=plt.gca(), color='red')

#%%
da_dem_all = xr.concat([da_dem21_2m, da_dem22_2m, da_dem23_2m, da_dem24, da_dem25_2m], dim='time')
da_dem_all['time'] = [2021, 2022, 2023, 2024, 2025]

## Count px where each year has valid data
# value 5 has valid data all years; use this as mask for dhdt 
mask_valid = xr.where(~np.isnan(da_dem_all), 1, 0).sum(dim='time') 
da_dhdt_2m = da_dem_all.diff(dim='time') ## this will give the dhdt for each year, at 2m resolution
da_dhdt_2m = da_dhdt_2m.where(mask_valid>4) ## keep only px where all 5 years have valid data (so where the sum of the mask is 4, since diff reduces the number of time steps by 1)
# da_dhdt_2m['time'] = '2021-2022', '2022-2023', '2023-2024', '2024-2025' ## assign time values to the dhdt dataarray

# ## calculate average annual dhdt for 2021-2025
da_dhdt_2125_2m = da_dhdt_2m.mean(dim='time') ## this will give the average annual dhdt for 2021-2025, at 2m resolution

da_dhdt_2125_2m.plot.imshow(vmin=-5, vmax=5, cmap='RdBu_r')
sarytor_outline.boundary.plot(ax=plt.gca(), color='red')


for da_dhdt, year in zip(da_dhdt_2m, ['2021-2022', '2022-2023', '2023-2024', '2024-2025']):
    fname = f'sarytor_dhdt_{year}_2m.tif'

    if not os.path.exists(os.path.join(path2data_clean, fname)):
        print(f'Saving {fname} to cleaned data directory...')
        da_dhdt.rio.to_raster(os.path.join(path2data_clean, fname))
    else:
        print(f"File {fname} already exists in cleaned data directory. Skipping save.")

fname = f'sarytor_dhdt_2021-2025_2m.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    print(f'Saving {fname} to cleaned data directory...')
    da_dhdt_2125_2m.rio.to_raster(os.path.join(path2data_clean, fname))
else:
    print(f"File {fname} already exists in cleaned data directory. Skipping save.")
    

#%%
''' ##################################
Bedrock
- thickness grid is from 2021, so use DEM 2021
- calculate as bed = DEM - thickness and bed=DEM where thickness==0

# ## thickness and DEM have different resolutions. 
CLEANED: do at lower, thickness, resolution of 25 m; as a bed of 2m resolution is probably less robust.
HOMOGEN: do at high 2m resolution

################################## '''

# da_dem = xr.open_dataarray(os.path.join(path2data_raw, 'sarytor_DEM_2021.tif')
#                              ).isel(band=0).drop_vars('band')
da_h = xr.open_dataarray(os.path.join(path2data_raw, 'sarytor_h_2021.tif')
                           ).isel(band=0).drop_vars('band')

assert da_dem21_2m.rio.crs == da_h.rio.crs, "CRS of DEM and thickness grid do not match"
# assert da_dem.rio.resolution() == da_h.rio.resolution(), "Resolution of DEM and thickness grid do not match"
# assert da_dem.shape == da_h.shape, "Shape of DEM and thickness grid do not match"
print(f"CRS of DEM and thickness grid match: {da_dem21_2m.rio.crs}")

print(da_dem21_2m.rio.resolution(), da_h.rio.resolution()) ## thickness has lower resolution, so upsample this one

# da_h_match = reproject_match_grid(da_dem, da_h, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan) ## from low to high res: nearest
da_dem_match = datafuncs.reproject_match_grid(da_h, da_dem21_2m, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan) ## from high to low res: bilinear

assert da_h.rio.resolution() == da_dem_match.rio.resolution(), "Resolution of DEM and thickness grid do not match"
assert da_dem_match.shape == da_h.shape, "Shape of DEM and thickness grid do not match"

## calculate difference, fill areas outside of glacier (where H is nan or 0) with DEM values (assuming bedrock = DEM there)
da_bedrock = da_dem_match - da_h
da_bedrock_filled = xr.where(np.isnan(da_bedrock), 
                               da_dem_match,  ## whre condition is ture
                               da_bedrock) ## where condition is false
da_bedrock_filled.plot.imshow()
sarytor_outline.boundary.plot(ax=plt.gca(), color='red')
## save to CLEAN directory
fname_bedr = 'sarytor_bedrock_25m.tif'
if not os.path.exists(os.path.join(path2data_clean, fname_bedr)):
    da_bedrock_filled.rio.to_raster(os.path.join(path2data_clean, fname_bedr))
else:
    print(f"File {fname_bedr} already exists in cleaned data directory. Skipping save.")
    

#%%
''' ##################################
Velocity
Provided: gappy velocity data and interpolate velocity data
CLEAN: both gappy and velocity data? --> use interpolated and also save FLAGS
HOMOGEN: the interpolated data
- provided at 26 m resolution
##################################
'''

da_vx_gappy = xr.open_dataarray(os.path.join(path2data_raw, 'sarytor_vx_2022-2023_gappy.tif')).isel(band=0).drop_vars('band')
da_vy_gappy = xr.open_dataarray(os.path.join(path2data_raw, 'sarytor_vy_2022-2023_gappy.tif')).isel(band=0).drop_vars('band')
da_vx_interp = xr.open_dataarray(os.path.join(path2data_raw, 'sarytor_vx_2022-2023.tif')).isel(band=0).drop_vars('band')
da_vy_interp = xr.open_dataarray(os.path.join(path2data_raw, 'sarytor_vy_2022-2023.tif')).isel(band=0).drop_vars('band')
assert da_vx_gappy.rio.crs == da_vy_gappy.rio.crs == da_vx_interp.rio.crs == da_vy_interp.rio.crs == 'EPSG:32644', "CRS of velocity grids do not match"
assert da_vx_gappy.rio.resolution() == da_vy_gappy.rio.resolution() == da_vx_interp.rio.resolution() == da_vy_interp.rio.resolution(), "Resolution of velocity grids do not match"

## plot fields
fig,axs=plt.subplots(2,2,figsize=(10,9))
da_vx_gappy.plot.imshow(ax=axs[0,0], cmap='PiYG',vmin=-10, vmax=10)
da_vy_gappy.plot.imshow(ax=axs[0,1], cmap='PiYG',vmin=-10, vmax=10)
da_vx_interp.plot.imshow(ax=axs[1,0], cmap='PiYG',vmin=-10, vmax=10)
da_vy_interp.plot.imshow(ax=axs[1,1], cmap='PiYG',vmin=-10, vmax=10)

## flag points where pixel has been interpolated or no
da_vx_flagged = xr.where(np.isnan(da_vx_gappy), 1, 0) ## 1 = valid, 0 = interpolated
da_vy_flagged = xr.where(np.isnan(da_vy_gappy), 1, 0) ## 1 = valid, 0 = interpolated
## check if vx_flagged and vy_flagged are the same
xr.testing.assert_equal(da_vx_flagged, da_vy_flagged) # raises assertionerror if not equal, so can continue script if nothing happens
da_v_flagged = da_vx_flagged.copy()

## save interpolated and flags to CLEANED dir
fname = 'sarytor_vx_2022-2023.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    da_vx_interp.rio.to_raster(os.path.join(path2data_clean, fname))

fname = 'sarytor_vy_2022-2023.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    da_vy_interp.rio.to_raster(os.path.join(path2data_clean, fname))

fname = 'sarytor_v-flagged.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    da_v_flagged.rio.to_raster(os.path.join(path2data_clean, fname))


#%% 


''' ##################################
Global uncertainties
################################## '''

'''# unct_thickness : unkown '''
'''unct_dhdt : unknown, approximately <= 0.5 '''
'''unct_velo : unknown, estimated at 10% ''' 

#%%
''' ##################################
HOMOGENIZED DATA
- fill all NaN values with 0
- do something else for DEM?
################################## '''

target_res = 2 # meter
target_crs = 'EPSG:32644'
# ## get bounds of DEM and thickness to get encompassing grid
# dem_bounds = da_dem_all.rio.bounds()
# h_bounds = da_h.rio.bounds()
# ## get minmax of both bounds, and add a buffer of 100 m to make sure all data is included in the homogenized grid
# sarytor_bounds = (min(dem_bounds[0], h_bounds[0]) - 100, min(dem_bounds[1], h_bounds[1]) - 100, 
#                   max(dem_bounds[2], h_bounds[2]) + 100, max(dem_bounds[3], h_bounds[3]) + 100)
da_dummy_target = datafuncs.create_regular_dummy_grid(da_dem_all, grid_res=target_res, crs=target_crs, unit='m', add_buffer=100)




#%%
''' ##################################
Make Elevation bins
--> 50 m binstep
--> based on earliest DEM if multiple available (assuming glacier is retreating, so earliest DEM has highest elevations)
--> do for both CLEAN and HOMOGNEIZED data; so possibly also resampling to different resolution
##################################
'''
da_dem_avg = da_dem_all.mean(dim='time').load() # is already at target res of 2m; use 'load' to mkae sure it's not a dask array anymore (gives error)

hmin = da_dem_avg.min().item()
hmax = da_dem_avg.max().item()

## for HOMOGENIZED: do not downsample elev-bin dataArray, but do new binning on donwsampled DEM
da_elev_bins, elev_bin_edges = datafuncs.dicretize_elevation_bins(da_dem_avg,
                                                     hmin=hmin, hmax=hmax,
                                                     binstep=50)

print('--- elev bins ---')
print(f'.. min max DEM: {np.round(hmin,0):.0f} to {np.round(hmax,0):.0f}')
print('.. bin edges: ', elev_bin_edges)

## save to CLEAN directory
fname = 'sarytor_elev-bins.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    da_elev_bins.rename('elevation_bins').rio.to_raster(os.path.join(path2data_clean, fname))

#%%
''' ##################################
Outlines to MASK
--> multi year available
--> make icemask
##################################
'''
# outline_mask_list = []
# for year_outline in [2021, 2023, 2025]: ## years with available outlines
#     print(year_outline)
gdf_outline1 = gpd.read_file(os.path.join(path2data_clean, f'sarytor_outline_2021.shp'))
da_icemask21 = (da_dummy_target * 2021).rio.clip(gdf_outline1.geometry, drop=False) # drop=False to keep the same grid and not drop the pixels outside the outline (which will be set to nodata)
gdf_outline2 = gpd.read_file(os.path.join(path2data_clean, f'sarytor_outline_2023.shp'))
da_icemask23 = (da_dummy_target * 2023).rio.clip(gdf_outline2.geometry, drop=False) # drop=False to keep the same grid and not drop the pixels outside the outline (which will be set to nodata)
gdf_outline3 = gpd.read_file(os.path.join(path2data_clean, f'sarytor_outline_2025.shp'))
da_icemask25 = (da_dummy_target * 2025).rio.clip(gdf_outline3.geometry, drop=False) # drop=False to keep the same grid and not drop the pixels outside the outline (which will be set to nodata)
# gdf_outline2 = gdf_2017; year2 = 2017

# da_outline_mask1 = (da_dummy_target*year1).rio.clip(gdf_outline1.geometry, gdf_outline1.crs, drop=False) # drop=False to keep the same grid and not drop the pixels outside the outline (which will be set to nodata)
# da_outline_mask2 = (da_dummy_target*year2).rio.clip(gdf_outline2.geometry, gdf_outline2.crs, drop=False) # drop=False to keep the same grid and not drop the pixels outside the outline (which will be set to nodata)
## combine to single dataArray
da_outline_mask = xr.concat([da_icemask21, da_icemask23, da_icemask25], 
                            dim='time').max(dim='time') 

#%%
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


## initial check that all variables have the same CRS, resolution and shape
## TO UPDATE 
da_var_dict = {'BED':da_bedrock.copy().rename('BED'),
                'DEM': da_dem_avg.copy().rename('DEM'),
                'ELEVBINS': da_elev_bins.copy().rename('ELEVBINS'),
                'THK': da_h.copy().rename('THK'),
                'DHDT': da_dhdt_2125_2m.copy().rename('DHDT'),
                'VX': da_vx_interp.copy().rename('VX'),
                'VY': da_vy_interp.copy().rename('VY'),
                'ICEMASK': da_outline_mask.copy().rename('ICEMASK'),
                # 'UNCT_VX': ##,
                # 'UNCT_VY': ##,
}
## resample to target grid where necessary
for varname , var in da_var_dict.items():
    print(varname)
    if var.rio.resolution()[0] != target_res:
        print(f'.. resampling {varname} from {var.rio.resolution()[0]} m to {target_res} m')
        var_target_res = datafuncs.reproject_match_grid(da_dummy_target, var, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)

        ## put back in dictionary
        da_var_dict[varname] = var_target_res
        print(var_target_res.rio.resolution(), var_target_res.shape)
    else: ## still resample, so I get no accidental geotransform issues from floating point precisin thingies
        print('.. updating coords')
        var_target_res = datafuncs.reproject_match_grid(da_dummy_target, var, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
        da_var_dict[varname] = var_target_res
    
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
                #    .rename('icemask')
                   .assign_attrs({'long_name':'Glacier Outline Mask',
                                  'units':'year',
                                  'uncertainty':'unknown',
                                  'crs':target_crs,
                                  'timestamp':'2021, 2023, 2025',
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
                .rename('DEM') 
                .assign_attrs({'long_name':'Elevation',
                                'units':'m',
                                'uncertainty':'unknown',
                                'crs':target_crs,
                                'timestamp':'2021-2025',
                                'description':'Average elevation data from annual DEMs between 2021-2025.'
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
                    .rename('BED')
                    .assign_attrs({'long_name':'Bedrock Elevation',
                                   'units':'m',
                                  'uncertainty':'unknown',
                                   'crs':target_crs,
                                   'timestamp':'2021',
                                   'description':'Bedrock elevation, calculated as DEM-thickness. Where thickness is 0, bedrock is set to DEM value.',
                                   })
                    # .drop_vars('spatial_ref')
                    .rio.write_crs(target_crs)
    )

# if da_elev_bins_hmg.isnull().any():
#     raise ValueError("Elevation bins has NaN values (since DEM has them), cannot assign nodata value of 0.")
# else: 
da_elev_bins_hmg = (da_var_dict['ELEVBINS'].copy()
                    # .fillna(-999) # fill NaN values with -999
                    .rename('ELEVBINS')
                    .assign_attrs({'long_name':'Elevation Bins',
                                  'units':'m',
                                  'uncertainty':'n/a',
                                  'crs':target_crs,
                                  'timestamp':'2021-2025',
                                  'description': f'Discretized elevation values into bins of 50 m. Using lowest (left-edge) value for each bin. Obtained from average DEM between 2021-2025.'
                                  })
                    .rio.write_crs(target_crs)
)

## thickness, dhdt, velo: can fill NaN with 0
da_thickness_hmg = (da_var_dict['THK'].copy()
                    .fillna(0)
                    .rename('THK')
                    .assign_attrs({'long_name':'Ice Thickness',
                                   'units':'m',
                                   'uncertainty':'unknown',
                                   'crs':target_crs,
                                   'timestamp':'2021',
                                   'description':'ice thickness interpolated from airborne GPR (UAV). Missing/NaN values were filled with 0.',
                                   'nodata': 0})
                .rio.write_crs(target_crs)
                    )
da_dhdt_hmg = (da_var_dict['DHDT'].copy()
               .fillna(0)
               .rename('DHDT')
               .assign_attrs({'long_name':'Surface Elevation Change',
                              'units':'m/year',
                              'uncertainty':'unknown, assumed to be <= 0.5 m',
                              'crs':target_crs,
                              'timestamp':'2021-2025',
                              'description':'Annual elevation change. Missing/NaN values were filled with 0.',
                              'nodata': 0})
                .rio.write_crs(target_crs)
               )

da_vx_hmg = (da_var_dict['VX'].copy()
             .fillna(0)
             .rename('VX')
             .assign_attrs({'long_name': 'Surface ice velocity (x-component)',
                            'units':'m/year',
                            'uncertainty':'unknown, estimated to be 10% of velocity value',
                            'crs':target_crs,
                            'timestamp':'2022-2023',
                            'description':'Velocity for the period 2022-2023. Missing/NaN values were filled with 0.',
                            'nodata': 0
                            })
                .rio.write_crs(target_crs)
)

da_vy_hmg = (da_var_dict['VY'].copy()
             .fillna(0)
             .rename('VY')
             .assign_attrs({'long_name': 'Surface ice velocity (y-component)',
                            'units':'m/year',
                            'uncertainty':'unknown, estimated to be 10% of velocity value',
                            'crs':target_crs,
                            'timestamp':'2022-2023',
                            'description':'Velocity for the period 2022-2023. Missing/NaN values were filled with 0.',
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

### final NaN check: no NaN values for within-glacier bounds allowed
for da in da_var_list:
    count_invalid = datafuncs.count_nan_values_in_glacier(da, gdf_outline1)
    if count_invalid > 0:
        print(f"Warning: There are still {count_invalid} NaN values in {da.name} after interpolation and filling.")



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

#%%
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

fname_nc = 'sarytor_glacier_observations.nc'

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


fname_nc = 'sarytor_glacier_observations.nc'

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
                                [(3800, 4800), (3800, 4800), (3800, 4800), (0,200),  (-5,5), (-20,20),(-20,20),None]):
    if vminmax is not None:
        vmin, vmax = vminmax
    else:
        vmin, vmax = None, None

    da_plot = ds_glacier_loaded[var]

    ax=axs[row,col]
    da_plot.plot.imshow(ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, cbar_kwargs={'shrink': 0.7})
    ax.set_title(var)
    col+=1
    if col >= 4:
        col = 0
        row += 1
[ax.set_aspect('equal') for ax in axs.flatten()];
[ax.set_axis_off() for ax in axs.flatten()];

fig.savefig(os.path.join(path2data_homog, 'sarytor_netcdf_vars.png'), dpi=300)
# %%

# %%
