#%% Check and pre-processing of data
''' Preprocessing of observational data for the Zongo glacier.
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

path2data_raw = '../../ContinuIX_WP1_data/Data_Package/01_submitted_data/Zongo/'
path2data_clean = '../../ContinuIX_WP1_data/Data_Package/02_raw-cleaned_data/Zongo/'
path2data_homog = '../../ContinuIX_WP1_data/Data_Package/03_homogenized_data/Zongo/'

import datafunctions as datafuncs

# #%% FUnctions --> to datafuncs

# def reproject_match_grid( ref_img_da, img_da , resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan):
#     ''' Match xarray grid of different spatial resolutions.'''
    
#     # Expected order: ('time', 'y', 'x')
#     dims = img_da.dims
#     if 'time' in dims:
#         ref_img_da = ref_img_da.transpose('time','y','x') # CRS is alreadyy written .rio.write_crs(3031, inplace=True)
#         img_da = img_da.transpose('time','y','x')
    
#     # -- reproject (even though same crs) and match grid (extent, resolution and projection)
#     img_repr_match = img_da.rio.reproject_match(ref_img_da,resampling=resample_method,nodata=nodata_value) # need to specify nodata, otherwise fills with (inf) number 1.79769313e+308

#     # advised to update coords to make the coordinates the exact same due to tiny differences in the coordinate values due to floating precision
#     img_repr_match = img_repr_match.assign_coords({
#         "y": ref_img_da.y,
#         "x": ref_img_da.x,
#     })
    
#     return img_repr_match.transpose(*dims) # transpose dimension order back to original


#%% Step 0: Check submitted (raw) data
''' ##################################
Check submitted (raw) data where needed


## DEMs: as submitted
## DHDT: is in meters --> to m/yr

## Velocity: 
- GRID: submitted millan velocity --> scale
- STAKES: 

## Thickness GRID
submitted millan velocity --> do not use; use GlaTE interpolated thickness (other script)


## Thickness profiles: 
- check csv, save as shp. Repojrect lat/lon to EPSG 32719

## Bedrock: not provided;
- Calculate from DEM - thickness
- Use GlaTE interpolated rather than millan thickness.
  Interpolated thickness is 2012, closest dem is 2013 (not 2006)

---
TO DO: check grid resolutions and homogenize these

##################################
'''
zongo_outline = gpd.read_file(os.path.join(path2data_raw, 'zongo_outline-2006.shp'))


''' ##################################
Thickness profiles
################################## '''

df_profiles = pd.read_csv(os.path.join(path2data_raw, 'zongo_h_profiles_20120809.csv'),
                         delimiter=';',
                         header=0)
df_profiles
df_profiles.head()
df_profiles['lat'] = df_profiles['latitude'].astype(float)
df_profiles['lon'] = df_profiles['longitude'].astype(float)

## convert to geodataframe, set geometry and CRS
crs_profiles = 'EPSG:4326'
gdf_profiles4326 = gpd.GeoDataFrame(df_profiles,
                                geometry=gpd.points_from_xy(df_profiles['lon'], df_profiles['lat']),
                                crs=crs_profiles)
## reproject to EPSG32719
gdf_profiles = gdf_profiles4326.to_crs('EPSG:32719')

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
fname_profiles = 'zongo_h_profiles_20120809.shp'
if not os.path.exists(os.path.join(path2data_clean, fname_profiles)):
    gdf_profiles.to_file(os.path.join(path2data_clean, fname_profiles))
else:
    print(f"File {fname_profiles} already exists in cleaned data directory. Skipping save.")


#%%
''' ##################################
Bedrock
################################## '''

da_dem_2006 = xr.open_dataarray(os.path.join(path2data_raw, 'zongo_DEM_2006.tif')
                    ).isel(band=0).drop_vars('band')
## dem2006 has 0 values instead of nan
da_dem_2006 = da_dem_2006.where(da_dem_2006 > 0, np.nan)
da_dem_2013 = xr.open_dataarray(os.path.join(path2data_raw, 'zongo_DEM_2013.tif')
                    ).isel(band=0).drop_vars('band')
da_h_2012 = xr.open_dataarray(os.path.join(path2data_clean, 'zongo_h_20120809.tif')
                    ).isel(band=0).drop_vars('band')
print(da_dem_2013.rio.crs)
assert da_dem_2013.rio.crs == da_h_2012.rio.crs, "CRS of DEM and thickness grid do not match"
assert da_dem_2013.rio.resolution() == da_h_2012.rio.resolution(), "Resolution of DEM and thickness grid do not match"
assert da_dem_2013.shape == da_h_2012.shape, "Shape of DEM and thickness grid do not match"

## calculate difference, fill areas outside of glacier (where H is nan or 0) with DEM values (assuming bedrock = DEM there)
da_bedrock = da_dem_2013 - da_h_2012
da_bedrock_filled = xr.where(np.isnan(da_bedrock), 
                               da_dem_2013,  ## whre condition is ture
                               da_bedrock) ## where condition is false
# da_bedrock_filled.plot.imshow()

fig,axs = plt.subplots(1,3, figsize=(15,5))
da_dem_2013.plot.imshow(ax=axs[0], vmin=4500, vmax=6000)
axs[0].set_title('DEM 2013')
da_h_2012.plot.imshow(ax=axs[1], vmin=0, vmax=200, cmap='Blues')
axs[1].set_title('Thickness 2012')  
da_bedrock_filled.plot.imshow(ax=axs[2], vmin=4500, vmax=6000)
axs[2].set_title('Bedrock (DEM - Thickness)')

## save to CLEAN directory
fname_bedr = 'zongo_bedrock.tif'
if not os.path.exists(os.path.join(path2data_clean, fname_bedr)):
    da_bedrock_filled.rio.to_raster(os.path.join(path2data_clean, fname_bedr))
else:
    print(f"File {fname_bedr} already exists in cleaned data directory. Skipping save.")

#%%
''' ##################################
dhdt
################################## '''

da_dh = xr.open_dataarray(os.path.join(path2data_raw, 'zongo_dh_2006-2013.tif')
                             ).isel(band=0).drop_vars('band')
## has bad NaN values .. mask lim values 
da_dh = da_dh.where(np.abs(da_dh) < 1000) ## set values with abs > 1000 to nan
da_dhdt = da_dh / (2013-2006) ## in m/yr

zongo_outline = gpd.read_file(os.path.join(path2data_raw, 'zongo_outline-2006.shp'))

assert da_dhdt.rio.crs == da_h_2012.rio.crs
assert da_dhdt.rio.resolution() == da_h_2012.rio.resolution()

## save to CLEAN directory
fname = 'zongo_dhdt_2006-2013.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    da_dhdt.rio.to_raster(os.path.join(path2data_clean, fname))
else:
    print(f"File {fname} already exists in cleaned data directory. Skipping save.")

# %%
''' ##################################
velocity fields:
take from Millan, 50m resolution.
- has been checked with stake measurements, seems to be OK; also checked Ducasse velocity. But since the Data Providers submitted the Millan velocity themselves, we favor this.
################################## '''

vx_millan = xr.open_dataarray(os.path.join(path2data_raw, 'zongo_vx_2017-2018.tif')
                             ).isel(band=0).drop_vars('band')
vy_millan = xr.open_dataarray(os.path.join(path2data_raw, 'zongo_vy_2017-2018.tif')
                             ).isel(band=0).drop_vars('band')

vx_std = xr.open_dataarray(os.path.join(path2data_raw, 'zongo_vx-std_2017-2018.tif')
                             ).isel(band=0).drop_vars('band')
vy_std = xr.open_dataarray(os.path.join(path2data_raw, 'zongo_vy-std_2017-2018.tif')
                             ).isel(band=0).drop_vars('band')

assert vx_millan.rio.crs == vy_millan.rio.crs == 'EPSG:32719'
print(vx_millan.rio.resolution())

## no further cleaning needed, save to CLEAN
fname = 'zongo_vx_2017-2018.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    vx_millan.rio.to_raster(os.path.join(path2data_clean, fname))
    vx_std.rio.to_raster(os.path.join(path2data_clean, 'zongo_vx-std_2017-2018.tif'))
else: print(f"File {fname} already exists in cleaned data directory. Skipping save.")
fname = 'zongo_vy_2017-2018.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    vy_millan.rio.to_raster(os.path.join(path2data_clean, fname))
    vy_std.rio.to_raster(os.path.join(path2data_clean, 'zongo_vy-std_2017-2018.tif'))
else: print(f"File {fname} already exists in cleaned data directory. Skipping save.")

#%%

''' ##################################
Global uncertainties
################################## '''

'''# unct_thickness : from readme '''
unct_thickness = 20 # m ; concerning in-situ profiles.
'''unct_dhdt : from readme '''
unct_dhdt = 2 # m
'''unct_velo : std from millan ''' 
unct_dem = 2 # m , from readme

#%%

def create_regular_dummy_grid(ds, grid_res, crs=None, unit='m', add_buffer=None):
    ''' Create a regular dummy grid with specified resolution and CRS, based on the extent of the input dataset/dataArray. 
    This can be used as a reference grid for reprojecting/matching other datasets. 
    Parameters:
    - ds: xarray Dataset or DataArray to define the extent of the grid
    - grid_res: desired grid resolution (in the same units as the CRS, e.g., meters)
    - crs: desired coordinate reference system (default: same as input dataset)
    - unit: unit of the grid resolution (default: 'm' for meters)
    '''
    if not crs:
        crs = ds.rio.crs
    x0 = ds.x.min().item() ; x1 = ds.x.max().item() ; y0 = ds.y.min().item() ; y1 = ds.y.max().item()
    if add_buffer:
        x0 -= add_buffer; x1 += add_buffer; y0 -= add_buffer; y1 += add_buffer
    ## make grid coordinates start and end at multiples of grid_res to avoid floating point precision issues
    x0 = np.floor(x0/grid_res)*grid_res; x1 = np.floor(x1/grid_res)*grid_res; 
    y0 = np.floor(y0/grid_res)*grid_res; y1 = np.floor(y1/grid_res)*grid_res
    x_seq = np.arange(x0, x1+grid_res, step=grid_res )
    y_seq = np.arange(y0, y1+grid_res, step=grid_res )

    ## check if y_seq is decreasing and reverse if needed
    if ds.rio.resolution()[1] < 0: # if y resolution is negative, then y_seq should be decreasing
        y_seq = y_seq[::-1]

    # ## get floating point presicion of grid_res, and apply that to x0 
    # # (e.g. if x0 is at 0.0730001 and grid_res is 0.08, then start xgrid at 287.00 instead of 287.0000000001)
    # decimal_places = int(-np.floor(np.log10(grid_res)))
    # ## round the sequences to avoid floating point precision issues (e.g. if x_seq is [287.00, 287.08, 287.16, ...] but due to floating point precision it is actually [287.0000000001, 287.0800000001, 287.1600000001, ...], then round to 2 decimal places to get rid of the tiny differences)
    # x_seq = np.round(x_seq, decimal_places)
    # y_seq = np.round(y_seq, decimal_places)

    grid_dummy = xr.DataArray(
        data=np.ones( (len(y_seq), len(x_seq)) ),
        dims=["y", "x"],
        coords=dict(
            y=y_seq,
            x=x_seq,
        ),
        attrs=dict(
            description=f"regular grid at {grid_res} {unit} resolution",
            unit=unit,
        ),
    ).rio.write_crs(crs)
    
    return grid_dummy

#%%
''' ##################################
HOMOGENIZED DATA
- fill all NaN values with 0
- do something else for DEM?
################################## '''

target_res = 25 # meter
target_crs = 'EPSG:32719'
# da_dummy_target = datafuncs.create_regular_dummy_grid(da_h_2012, grid_res=target_res, crs=target_crs, unit='m', add_buffer=100)
da_dummy_target = create_regular_dummy_grid(da_h_2012, grid_res=target_res, crs=target_crs, unit='m', add_buffer=100)

print(da_dummy_target.rio.resolution() )
## check bounds by simple plot
# da_dummy_target.plot.imshow()
# zongo_outline.boundary.plot(ax=plt.gca())




#%%
''' ##################################
Make Elevation bins
--> 50 m binstep
--> based on earliest DEM if multiple available (assuming glacier is retreating, so earliest DEM has highest elevations)
--> do for both CLEAN and HOMOGNEIZED data; so possibly also resampling to different resolution
##################################
'''
da_dem_avg = xr.concat([da_dem_2006, da_dem_2013], dim='time').mean(dim='time').load() # is already at target res of 2m; use 'load' to mkae sure it's not a dask array anymore (gives error)

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
fname = 'zongo_elev-bins.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    da_elev_bins.rename('elevation_bins').rio.to_raster(os.path.join(path2data_clean, fname))

#%%
''' ##################################
Outlines to MASK
--> multi year available
--> make icemask
##################################
'''

gdf_outline1 = gpd.read_file(os.path.join(path2data_clean, 'zongo_outline-2006.shp'))
gdf_outline2 = gpd.read_file(os.path.join(path2data_clean, 'zongo_outline-2013.shp'))
da_icemask1 = (da_dummy_target * 2006).rio.clip(gdf_outline1.geometry, drop=False) 
da_icemask2 = (da_dummy_target * 2013).rio.clip(gdf_outline2.geometry, drop=False)
# gdf_outline2 = gdf_2017; year2 = 2017

# da_outline_mask1 = (da_dummy_target*year1).rio.clip(gdf_outline1.geometry, gdf_outline1.crs, drop=False) # drop=False to keep the same grid and not drop the pixels outside the outline (which will be set to nodata)
# da_outline_mask2 = (da_dummy_target*year2).rio.clip(gdf_outline2.geometry, gdf_outline2.crs, drop=False) # drop=False to keep the same grid and not drop the pixels outside the outline (which will be set to nodata)
## combine to single dataArray
da_outline_mask = xr.concat([da_icemask1, da_icemask2], 
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
da_var_dict = {'BED':da_bedrock_filled.copy().rename('BED'),
                'DEM': da_dem_avg.copy().rename('DEM'),
                'ELEVBINS': da_elev_bins.copy().rename('ELEVBINS'),
                'THK': da_h_2012.copy().rename('THK'),
                'DHDT': da_dhdt.copy().rename('DHDT'),
                'VX': vx_millan.copy().rename('VX'),
                'VY': vy_millan.copy().rename('VY'),
                'ICEMASK': da_outline_mask.copy().rename('ICEMASK'),
                'UNCT_VX': vx_std.copy().rename('UNCT_VX'),
                'UNCT_VY': vy_std.copy().rename('UNCT_VY'),
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
                                  'uncertainty':'n/a',
                                  'crs':target_crs,
                                  'timestamp':'2006, 2013',
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
                                'uncertainty':f'+- {unct_dem} m',
                                'crs':target_crs,
                                'timestamp':'2006-2013',
                                'description':'Average elevation data from annual DEMs between 2006-2013.'
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
                                   'timestamp':'2013',
                                   'description':'Bedrock elevation, calculated as DEM (2013) - thickness (2012). Where thickness is 0, bedrock is set to DEM value.',
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
                                  'uncertainty': 'n/a',
                                  'crs':target_crs,
                                  'timestamp':'2006-2013',
                                  'description': f'Discretized elevation values into bins of 50 m. Using lowest (left-edge) value for each bin. Obtained from average DEM between 2006-2013.'
                                  })
                    .rio.write_crs(target_crs)
)

## thickness, dhdt, velo: can fill NaN with 0
da_thickness_hmg = (da_var_dict['THK'].copy()
                    .fillna(0)
                    # .rename('thickness')
                    .assign_attrs({'long_name':'Ice Thickness',
                                   'units':'m',
                                   'uncertainty':f'+- {unct_thickness} m',
                                   'crs':target_crs,
                                   'timestamp':'2013',
                                   'description':'ice thickness interpolated from airborne GPR (UAV). Missing/NaN values were filled with 0.',
                                   'nodata': 0})
                .rio.write_crs(target_crs)
                    )
da_dhdt_hmg = (da_var_dict['DHDT'].copy()
               .fillna(0)
            #    .rename('dhdt')
               .assign_attrs({'long_name':'Surface Elevation Change',
                              'units':'m/year',
                              'uncertainty':f'+- {unct_dhdt} m',
                              'crs':target_crs,
                              'timestamp':'2006-2013',
                              'description':'Annual elevation change. Missing/NaN values were filled with 0.',
                              'nodata': 0})
                .rio.write_crs(target_crs)
               )

da_vx_hmg = (da_var_dict['VX'].copy()
             .fillna(0)
            #  .rename('vx')
             .assign_attrs({'long_name': 'Surface ice velocity (x-component)',
                            'units':'m/year',
                            'uncertainty': 'provided as grid, UNCT_VX',
                            'crs':target_crs,
                            'timestamp':'2017-2018',
                            'description':'Velocity for the period 2017-2018. Missing/NaN values were filled with 0.',
                            'nodata': 0
                            })
                .rio.write_crs(target_crs)
)

da_vy_hmg = (da_var_dict['VY'].copy()
             .fillna(0)
            #  .rename('vy')
             .assign_attrs({'long_name': 'Surface ice velocity (y-component)',
                            'units':'m/year',
                            'uncertainty': 'provided as grid, UNCT_VY',
                            'crs':target_crs,
                            'timestamp':'2017-2018',
                            'description':'Velocity for the period 2017-2018. Missing/NaN values were filled with 0.',
                            'nodata': 0
                            })
                .rio.write_crs(target_crs)
)

da_vx_std_hmg = (da_var_dict['UNCT_VX'].copy()
             .fillna(0)
            #  .rename('vx_std')
             .assign_attrs({'long_name': 'Uncertainty of surface ice velocity (x-component)',
                            'units':'m/year',
                            'uncertainty': 'n/a',
                            'crs':target_crs,
                            'timestamp':'2017-2018',
                            'description':'Standard deviation of velocity for the period 2017-2018. Missing/NaN values were filled with 0.',
                            'nodata': 0
                            })
                .rio.write_crs(target_crs)
)

da_vy_std_hmg = (da_var_dict['UNCT_VY'].copy()
             .fillna(0)
            #  .rename('vy_std')
             .assign_attrs({'long_name': 'Uncertainty of surface ice velocity (y-component)',
                            'units':'m/year',
                            'uncertainty': 'n/a',
                            'crs':target_crs,
                            'timestamp':'2017-2018',
                            'description':'Standard deviation of velocity for the period 2017-2018. Missing/NaN values were filled with 0.',
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
                da_vx_std_hmg,
                da_vy_std_hmg
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

## check for NaN values:
# da_bedrock_hmg.plot.imshow()
# gdf_outline1.boundary.plot(ax=plt.gca())

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

fname_nc = 'zongo_glacier_observations.nc'

try:
    print(f'--> saving homogenized data to netcdf "{fname_nc}"; overwriting if file exists')
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

fname_nc = 'zongo_glacier_observations.nc'

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
                                [(4800,6000), (4800,6000), (4800,6000), (0,200),  (-5,5), (-20,20),(-20,20), None]):
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

fig.savefig(os.path.join(path2data_homog, 'zongo_netcdf_vars.png'), dpi=300)
# %%

# %%