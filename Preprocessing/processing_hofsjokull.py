#%% Check and pre-processing of data
''' Preprocessing of observational data for the Hofsjokull glacier.
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

path2data_raw = '../../ContinuIX_WP1_data/Data_Package/01_submitted_data/Hofsjokull/'
path2data_clean = '../../ContinuIX_WP1_data/Data_Package/02_raw-cleaned_data/Hofsjokull/'
path2data_homog = '../../ContinuIX_WP1_data/Data_Package/03_homogenized_data/Hofsjokull/'

import datafunctions as datafuncs



#%% Step 0: Check submitted (raw) data
''' ##################################
Check submitted (raw) data where needed


## DEMs: as submitted
## DHDT: calculate from DEMs
- individual dhdt periods --> to cleaned data folder
- average dhdt --> to homogenized data folder (also in target resolution)

## Velocity: gappy data; to interpolate/average

## Thickness GRID: not provided,
calculate from DEM YYYY and BED (which most represents 2013, but assuming constant).

## Thickness profiles: not available

## Bedrock: as submitted.

## Outlines: from elevaiton-lidar survey in 2013
- combine with flow basin infromation from RGIv6

---
TO DO: check grid resolutions and homogenize these

##################################
'''
# target_res = 100 # m; target resolution for homogenized data
target_crs = 'EPSG:3057' # Icelandic projection system

''' ##################################
Outlines  
################################## '''
gdf_rgiv6 = gpd.read_file('../../ContinuIX_WP1_data/other_data/outline_Hofsjokull_RGIv6_4326.shp')
# gdf_rgiv6.plot()
print(gdf_rgiv6.crs)
## reproject to iceland reprojection system
gdf_hofsj = gdf_rgiv6.to_crs(3057)

# ## save to cleaned dir
# fname = 'hofsjokull_outline_RGIv6.shp'
# if not os.path.exists(os.path.join(path2data_clean, fname)):
#     gdf_hofsj.to_file(os.path.join(path2data_clean, fname))
# else:
#     print(f"File {fname} already exists in cleaned data directory. Skipping save.")

## FOR PLOTS in this script
## merge all geometries into one (dissolve) to have a single outline for clipping
hofsj_union = gdf_hofsj.geometry.union_all()   # GeoPandas/Shapely recent versions
# or (older API): hofsj_union = gdf_hofsj.unary_union
gdf_hofsj_union = gpd.GeoDataFrame(geometry=[hofsj_union], crs=gdf_hofsj.crs)

'''## OUTLINE 2013: 
form lidar survey with the bedrock results in areas that have emerged from glacier melting
--> can use these as glacier outlines for 2013 '''

elev_line_datfile = os.path.join('../../ContinuIX_WP1_data/06_Hofsjokull/Hofsjokull_bedrock/', 'Hofs-outline_2013-elevation-fp.dat')
gdf_outline_elev = pd.read_csv(elev_line_datfile)
# gdf_outline_elev = gpd.GeoDataFrame(gdf_outline_elev, geometry=gpd.points_from_xy(gdf_outline_elev['x-i93'], gdf_outline_elev['y-i93']), crs=target_crs)
gdf_outline_elev.rename(columns={'x-i93':'x', 'y-i93':'y'}, inplace=True)
## this is a gataframe with many points, so now merge to a single polygon
## make a polygon from all points, 
from shapely.geometry import Polygon
coords = np.column_stack([gdf_outline_elev.x, gdf_outline_elev.y])
poly = Polygon(coords)

gdf_outline_2013 = gpd.GeoDataFrame(geometry=[poly], crs=target_crs)
gdf_outline_2013["geometry"] = gdf_outline_2013.geometry.make_valid()

### combine RGI6 basins to the 2013 outline 


# Clip each flow basin to the ice-cap outline
icecap_geom = gdf_outline_2013.geometry.item()  

gdf_hofsj_clipped = gdf_hofsj.copy()
gdf_hofsj_clipped["geometry"] = gdf_hofsj_clipped.geometry.intersection(icecap_geom)

# Remove basins that do not overlap the ice cap
gdf_hofsj_clipped = gdf_hofsj_clipped[
    ~gdf_hofsj_clipped.geometry.is_empty
].copy()
# fig,ax=plt.subplots(1,1,figsize=(8,8))
# gdf_outline_2013.plot(ax=ax)
# gdf_hofsj.boundary.plot(ax=ax, color='red')
# gdf_hofsj_clipped.boundary.plot(ax=ax, color='black')

## save outline to CLEANED dir
gdf_basins_2013 = gdf_hofsj_clipped[['RGIId','GLIMSId','CenLon','CenLat','geometry']].copy()

fname = 'hofsjokull_outline_2013.shp'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    gdf_basins_2013.to_file(os.path.join(path2data_clean, fname))




#%%
''' ##################################
dhdt  
################################## '''

da_dem_2013 = xr.open_dataarray(os.path.join(path2data_raw, 'hofsjokull_dem_20131013.tif')
                             ).isel(band=0).drop_vars('band')
da_dem_2020 = xr.open_dataarray(os.path.join(path2data_raw, 'hofsjokull_dem_202010.tif')
                             ).isel(band=0).drop_vars('band')
da_dem_2023 = xr.open_dataarray(os.path.join(path2data_raw, 'hofsjokull_dem_202309.tif')
                             ).isel(band=0).drop_vars('band')

assert da_dem_2013.rio.crs == da_dem_2020.rio.crs == da_dem_2023.rio.crs, "CRS should be the same for all DEM files"
assert da_dem_2013.rio.resolution() == da_dem_2020.rio.resolution() == da_dem_2023.rio.resolution(), "Resolution should be the same for all DEM files"
# assert da_dem_2013.shape == da_dem_2020.shape == da_dem_2023.shape, "Shape should be the same for all DEM files"
print('CRS of Hofsjokull DEM files:', da_dem_2013.rio.crs)

## 2023 DEM file has slightly different grid boundaries: reproject match
da_dem_2023 = datafuncs.reproject_match_grid(da_dem_2020, da_dem_2023, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)


ds_DEM = xr.concat([da_dem_2013, da_dem_2020, da_dem_2023], dim='time').assign_coords(time=[2013, 2020, 2023])

#%%
## calculate elevation change for different intervals and average

da_dhdt_1320 =(da_dem_2020 - da_dem_2013) / (2020-2013)
da_dhdt_1323 = (da_dem_2023 - da_dem_2013) / (2023-2013)
da_dhdt_2023 = (da_dem_2023 - da_dem_2020) / (2023-2020)
# da_dhdt_all = xr.concat([da_dhdt_1320, da_dhdt_1323, da_dhdt_2023], dim='time').assign_coords(time=['2013-2020', '2013-2023', '2020-2023'])
 
## dhdt 13-23 should be similar to dhdt1320+dhdt2023: check
# da_dhdt_1323_b = (da_dhdt_1320*7+da_dhdt_2023*3)/10

## each dhdt period
# fig,axs=plt.subplots(1,3, figsize=(17,5))
# da_dhdt_1323.rename('dhdt (m/yr)').plot.imshow(ax=axs[0], vmin=-4, vmax=4, cmap="RdBu_r")
# axs[0].set_title('2013 - 2023')
# da_dhdt_1320.rename('dhdt (m/yr)').plot.imshow(ax=axs[1], vmin=-4, vmax=4, cmap="RdBu_r")
# axs[1].set_title('2013 - 2020')
# da_dhdt_2023.rename('dhdt (m/yr)').plot.imshow(ax=axs[2], vmin=-4, vmax=4, cmap="RdBu_r")
# axs[2].set_title('2020 - 2023')

## average dhdts comparison
# fig,axs=plt.subplots(1,2, figsize=(12,5))
# da_dhdt_1323.rename('dhdt (m/yr)').plot.imshow(ax=axs[0], vmin=-4, vmax=4, cmap="RdBu_r")
# axs[0].set_title('dH/dt: DEM 2023 - DEM 2013')
# da_dhdt_1323_b.rename('dhdt (m/yr)').plot.imshow(ax=axs[1], vmin=-4, vmax=4, cmap="RdBu_r")
# axs[1].set_title('dH/dt: (dH/dt 13-20 + dH/dt 20-23)/2')

## save dhdt periods
fname1323 = f'hofsjokull_dhdt_2013-2023.tif'
fname1320 = f'hofsjokull_dhdt_2013-2020.tif'
fname2023 = f'hofsjokull_dhdt_2020-2023.tif'
if not os.path.exists(os.path.join(path2data_clean, fname1323)):
    print(f'Saving dhdt to cleaned data directory...')
    da_dhdt_1320.rio.to_raster(os.path.join(path2data_clean, fname1320))
    da_dhdt_1323.rio.to_raster(os.path.join(path2data_clean, fname1323))
    da_dhdt_2023.rio.to_raster(os.path.join(path2data_clean, fname2023))
else:
    print(f"File {fname1323} already exists in cleaned data directory. Skipping save.")


#%% Load iceland geoid info

''' ##################################
load GEOID
correct DEMs from ELLIPSOID to ORTHOMTETRIC heights
################################## '''
geoid_file = '../../ContinuIX_WP1_data/other_data/iceland_geoid.txt'

# -------------------------
# Read header
# -------------------------
header = {}
data_start_line = None

with open(geoid_file, "r") as f:
    for i, line in enumerate(f):
        line = line.strip()
        
        if line.startswith("begin_of_head"):
            data_start_line = i + 1
            continue
        if line.startswith("end_of_head"):
            data_start_line = i + 1
            break
        if "=" in line:
            key, value = line.split("=", 1)
            header[key.strip().replace(" ", "_")] = np.float32(value.strip())
        elif ":" in line:
            key, value = line.split(":", 1)
            header[key.strip().replace(" ", "_")] = value.strip()
header['ncols'] = int(header['ncols'])
header['nrows'] = int(header['nrows'])

# -------------------------
# Read data values
# -------------------------
data = np.loadtxt(geoid_file, skiprows=data_start_line)
assert data.shape[0] == header['nrows'] and data.shape[1] == header['ncols'], f"Data shape {data.shape} does not match header dimensions ({header['nrows']}, {header['ncols']})"

# Replace nodata with NaN
geoid = np.where(data == header['nodata'], np.nan, data)

## construct dataArray
grid_res_lon = header['delta_lon']
grid_res_lat = header['delta_lat']
lon_coords = np.arange(header['lon_min'], header['lon_max'], grid_res_lon)
lat_coords = np.arange(header['lat_min'], header['lat_max'], grid_res_lat)
## flip latitude coordinates to be in descending order (from north to south)
lat_coords = lat_coords[::-1]
assert len(lon_coords) == header['ncols'], f"Longitude coordinates length {len(lon_coords)} does not match ncols {header['ncols']}"
assert len(lat_coords) == header['nrows'], f"Latitude coordinates length {len(lat_coords)} does not match nrows {header['nrows']}"
da_geoid_iceland = xr.DataArray(
                        geoid,
                        dims=("y", "x"),
                        coords={
                            "y": lat_coords,
                            "x": lon_coords,
                        },
                        name="geoid_height",
                        attrs={
                            "long_name": "Geoid height",
                            "units": "m",
                            "model": header['model_name'],
                        })

## set raster information
da_geoid_iceland.rio.write_crs("EPSG:4326", inplace=True)  # Assuming the geoid data is in WGS84

## reproject coords to a resolution of 500 m (still manageble, will increase resolution to match DEM later)
da_geoid_iceland_3057 = da_geoid_iceland.rio.reproject(target_crs, resolution=500, resampling=rio.enums.Resampling.bilinear)

## now get geoid only for hofsjokull
hofsj_extent = da_dem_2013.rio.bounds()
da_geoid_hofsj_3057 = da_geoid_iceland_3057.rio.clip_box(*hofsj_extent, crs=target_crs)

#%%
''' Correct DEMs from ELLIPSOID to ORTHOMETRIC heights  '''
## resample to high grid
da_geoid_hofsj_2m = datafuncs.reproject_match_grid(da_dem_2013, da_geoid_hofsj_3057, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)
ds_DEM_ortho = ds_DEM - da_geoid_hofsj_2m

# #%% plot DEM comparison
# fig,axs=plt.subplots(3,3,figsize=(16,14))
# col=0
# for year in [2013, 2020, 2023]:
#     ax=axs[0,col]
#     ds_DEM.sel(time=year).isel(x=slice(7000,8000), y=slice(7000,8000)
#                                ).plot.imshow(ax=ax, cmap='terrain', vmin=1000, vmax=2000)
#     ax.set_title(f'DEM {year} (ellipsoid)')
#     ax=axs[1,col]
#     ds_DEM_ortho.sel(time=year).isel(x=slice(7000,8000), y=slice(7000,8000)).plot.imshow(ax=ax, cmap='terrain', vmin=1000, vmax=2000)
#     ax.set_title(f'DEM {year} (orthometric)')
#     ax=axs[2,col]
#     (ds_DEM_ortho.sel(time=year)-ds_DEM.sel(time=year)
#             ).isel(x=slice(7000,8000), y=slice(7000,8000)
#                    ).plot.imshow(ax=ax, cmap='Reds_r', )#vmin=-70, vmax=-60)
#     ax.set_title(f'Diff {year} (orthometric - ellipsoid)')
#     col += 1
    


#%% Thickness
''' ##################################
Thickness 
################################## '''

da_bed = xr.open_dataarray(os.path.join(path2data_raw, 'hofsjokull_bedrock_2013.tif')
                             ).isel(band=0).drop_vars('band')

print(da_bed.rio.crs, da_dem_2013.rio.crs) ## both are 3057 but still get assertion error for some reason; manually update crs
da_bed.rio.write_crs(3057,inplace=True)
assert da_bed.rio.crs == da_dem_2013.rio.crs, "CRS should be the same for all DEM files"

## interpolate bed resolution (200m) to DEM resolution (2m)
## no: do both to 25m; since otherwise thickness files are unnecesarily large.

### calculate thickness for every year, save file
da_dummy_25m = datafuncs.create_regular_dummy_grid(da_dem_2013, 
                                                      grid_res=25, 
                                                      crs=target_crs, unit='m')

da_bed_25m = datafuncs.reproject_match_grid(da_dummy_25m, da_bed, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)


# for da_dem, year in zip([da_dem_2013, da_dem_2020, da_dem_2023], ['2013', '2020', '2023']):
for year in ['2013', '2020', '2023']:
    da_dem = ds_DEM_ortho.sel(time=int(year)) ## update to get thickness with ortho DEM
    da_dem_25m = datafuncs.reproject_match_grid(da_dummy_25m, da_dem, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
    
    ## calculate thickness, remove any negative thickness values 
    da_thickness = da_dem_25m - da_bed_25m
    da_thickness = da_thickness.where(da_thickness>0, other=np.nan)

    fname = f'hofsjokull_h_{year}.tif'
    if not os.path.exists(os.path.join(path2data_clean, fname)):
        print(f'Saving {fname} to cleaned data dir')
        da_thickness.rio.to_raster(os.path.join(path2data_clean, fname))
    else:
        print(f"File {fname} already exists in cleaned data directory. Skipping save.")



#%%
''' ------------------
### HOFSJOKUL VELOCITIES  
----------------------'''
## 100 m resolution; in METER / DAY; need to convert to m/yr by multiplying by 365.25
data_dir = '../../ContinuIX_WP1_data/'
filelist_velo_hofsj = sorted([f for f in os.listdir(os.path.join(data_dir,'06_Hofsjokull/Hofsjokull_velocities/')) if f.endswith('.tif')])
filelist_velo_vz = [file for file in filelist_velo_hofsj if 'vertical' in file] # vertical velocity
years_vz = [2014, 2015, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
## get easting and northign
filelist_velo_hofsj = [file for file in filelist_velo_hofsj if 'easting' in file or 'northing' in file] # exclude stdev files
filelist_velo_std = [file for file in filelist_velo_hofsj if 'stddev' in file] # get stdev files for later use
filelist_vx_std = [file for file in filelist_velo_std if 'easting' in file]
filelist_vy_std = [file for file in filelist_velo_std if 'northing' in file]

filelist_velo_hofsj = [file for file in filelist_velo_hofsj if 'stddev' not in file] # exclude stdev files 
filelist_vx = [file for file in filelist_velo_hofsj if 'easting' in file]
filelist_vy = [file for file in filelist_velo_hofsj if 'northing' in file]
years_vx = [2014,2016,2017,     2019,2020,2021,2022,2023] ## left year (2014-2015 noted as 2014)
years_vy = [2014,2016,2017,2018,2019,2020,2021,2022,2023]
## drop 2018 from filelist 
filelist_vx = [f for f in filelist_vx if 's2018' not in f]
filelist_vy = [f for f in filelist_vy if 's2018' not in f]
filelist_vx_std = [f for f in filelist_vx_std if 's2018' not in f]
filelist_vy_std = [f for f in filelist_vy_std if 's2018' not in f]
years_vy = [2014,2016,2017,     2019,2020,2021,2022,2023] ## update

assert len(filelist_vx) == len(filelist_vy)
assert len(filelist_vx_std) == len(filelist_vy_std)

## open dataset
ds_hofsj_vx = xr.open_mfdataset([os.path.join(data_dir,'06_Hofsjokull/Hofsjokull_velocities/', f) for f in filelist_vx], 
                                engine='rasterio', join='outer', combine='nested', concat_dim='time'
                                ).assign_coords(time=years_vx).isel(band=0).drop_vars('band').rename({'band_data':'vx'})
ds_hofsj_vy = xr.open_mfdataset([os.path.join(data_dir,'06_Hofsjokull/Hofsjokull_velocities/', f) for f in filelist_vy],
                                 engine='rasterio', join='outer', combine='nested', concat_dim='time'
                                 ).assign_coords(time=years_vy).isel(band=0).drop_vars('band').rename({'band_data':'vy'})

ds_hofsj_vx_std = xr.open_mfdataset([os.path.join(data_dir,'06_Hofsjokull/Hofsjokull_velocities/', f) for f in filelist_vx_std],
                                 engine='rasterio', join='outer', combine='nested', concat_dim='time'
                                 ).assign_coords(time=years_vx).isel(band=0).drop_vars('band').rename({'band_data':'vx'})
ds_hofsj_vy_std = xr.open_mfdataset([os.path.join(data_dir,'06_Hofsjokull/Hofsjokull_velocities/', f) for f in filelist_vy_std],
                                 engine='rasterio', join='outer', combine='nested', concat_dim='time'
                                 ).assign_coords(time=years_vy).isel(band=0).drop_vars('band').rename({'band_data':'vy'})
ds_hofsj_vz = xr.open_mfdataset([os.path.join(data_dir,'06_Hofsjokull/Hofsjokull_velocities/', f) for f in filelist_velo_vz],
                                 engine='rasterio', join='outer', combine='nested', concat_dim='time'
                                 ).assign_coords(time=years_vz).isel(band=0).drop_vars('band').rename({'band_data':'vz'})
assert ds_hofsj_vx.rio.crs == ds_hofsj_vy.rio.crs, f"CRS should match between vx and vy; are {ds_hofsj_vx.rio.crs} and {ds_hofsj_vy.rio.crs}"

print('CRS of Hofsjokull velocity files:', ds_hofsj_vx.rio.crs)


## save each year to RAW datafolder with renaming of the file BEFORE CONVERTING TO m/yr
for year in years_vy:
    da_vx = ds_hofsj_vx['vx'].sel(time=year)
    da_vy = ds_hofsj_vy['vy'].sel(time=year)
    fname_vx = f'hofsjokull_vx_{year}-{year+1}_mday.tif'
    fname_vy = f'hofsjokull_vy_{year}-{year+1}_mday.tif'

    da_vx_std = ds_hofsj_vx_std['vx'].sel(time=year)
    da_vy_std = ds_hofsj_vy_std['vy'].sel(time=year)
    fname_vx_std = f'hofsjokull_vx_{year}-{year+1}_std_mday.tif'
    fname_vy_std = f'hofsjokull_vy_{year}-{year+1}_std_mday.tif'

    print(fname_vx)
    ## VX and VY
    if not os.path.exists(os.path.join(path2data_raw, fname_vx)):
        print(f'Saving {fname_vx} to RAW data dir')
        da_vx.rio.to_raster(os.path.join(path2data_raw, fname_vx))
    else:
        print(f"File {fname_vx} already exists in cleaned data directory. Skipping save.")
    if not os.path.exists(os.path.join(path2data_raw, fname_vy)):
        print(f'Saving {fname_vy} to cleaned data dir')
        da_vy.rio.to_raster(os.path.join(path2data_raw, fname_vy))
    else:
        print(f"File {fname_vy} already exists in cleaned data directory. Skipping save.")
    
    ## STDEVS
    if not os.path.exists(os.path.join(path2data_raw, fname_vx_std)):
        print(f'Saving {fname_vx_std} to RAW data dir')
        da_vx_std.rio.to_raster(os.path.join(path2data_raw, fname_vx_std))
    if not os.path.exists(os.path.join(path2data_raw, fname_vy_std)):
        print(f'Saving {fname_vy_std} to RAW data dir')
        da_vy_std.rio.to_raster(os.path.join(path2data_raw, fname_vy_std))

    ## VERTICAL
    if year in years_vz:
        da_vz = ds_hofsj_vz['vz'].sel(time=year)
        fname_vz = f'hofsjokull_v-vertical_{year}-{year+1}_mday.tif'
        if not os.path.exists(os.path.join(path2data_raw, fname_vz)):
            print(f'Saving {fname_vz} to RAW data dir')
            da_vz.rio.to_raster(os.path.join(path2data_raw, fname_vz))

## convert m/day to m/year
ds_hofsj_vx['vx'] = ds_hofsj_vx['vx'] * 365.25
ds_hofsj_vy['vy'] = ds_hofsj_vy['vy'] * 365.25

ds_hofsj_vx_std['vx'] = ds_hofsj_vx_std['vx'] * 365.25
ds_hofsj_vy_std['vy'] = ds_hofsj_vy_std['vy'] * 365.25
ds_hofsj_vz['vz'] = ds_hofsj_vz['vz'] * 365.25

## update attribute 
ds_hofsj_vx['vx'].attrs['units'] = 'm/yr'
ds_hofsj_vy['vy'].attrs['units'] = 'm/yr'
ds_hofsj_vz['vz'].attrs['units'] = 'm/yr'
ds_hofsj_vx_std['vx'].attrs['units'] = 'm/yr'   
ds_hofsj_vy_std['vy'].attrs['units'] = 'm/yr'


# '''## plot velocity components for all years'''
# ds_hofsj_vx['vx'].plot.imshow(cmap='PiYG', col='time', col_wrap=2, vmin=-50, vmax=50, cbar_kwargs={'fraction':0.02, 'label':'Velocity (m/yr)'})
# ## get axes
# axs = plt.gcf().axes[:-1] # exclude colorbar axis
# [gdf_hofsj_union.boundary.plot(ax=ax,linestyle='--', color='white', linewidth=1) for ax in axs]
# [ax.set_axis_off() for ax in axs]

# ## plot velocity components for all years
# ds_hofsj_vy['vy'].plot.imshow(cmap='PiYG', col='time', col_wrap=2, vmin=-50, vmax=50, cbar_kwargs={'fraction':0.02, 'label':'Velocity (m/yr)'})
# ## get axes
# axs = plt.gcf().axes[:-1] # exclude colorbar axis
# [gdf_hofsj_union.boundary.plot(ax=ax,linestyle='--', color='white', linewidth=1) for ax in axs]
# [ax.set_axis_off() for ax in axs]

# '''## calculate velocity magnitude'''
# da_hofsj_v = (np.sqrt(ds_hofsj_vx['vx']**2 + ds_hofsj_vy['vy']**2)).rename('velocity')

# ## plot velocity for all years
# da_hofsj_v.plot.imshow(cmap='viridis', col='time', col_wrap=2, vmin=0, vmax=100, cbar_kwargs={'fraction':0.02, 'label':'Velocity (m/yr)'})
# ## get axes
# axs = plt.gcf().axes[:-1] # exclude colorbar axis
# [gdf_hofsj_union.boundary.plot(ax=ax,linestyle='--', color='white', linewidth=1) for ax in axs]
# [ax.set_axis_off() for ax in axs]

#%% Select only 2017-18, 2019-20, 2020-21, 2023-24
years_select = [2017,2019,2020,2023]
years_select = [2017,2019,2020] ## update: not 2023, as its outside DEM bounds

da_hofsj_vx_sel = ds_hofsj_vx.sel(time=years_select)['vx']
da_hofsj_vy_sel = ds_hofsj_vy.sel(time=years_select)['vy']
da_hofsj_v_sel = (np.sqrt(da_hofsj_vx_sel**2 + da_hofsj_vy_sel**2)).rename('velocity')

# ### Save AVERAGE to CLEAN folder
# fname_vx = f'hofsjokull_vx_{years_select[0]}-{years_select[-1]}.tif'
# fname_vy = f'hofsjokull_vy_{years_select[0]}-{years_select[-1]}.tif'
# if not os.path.exists(os.path.join(path2data_clean, fname_vx)):
#     print(f'Saving {fname_vx} to cleaned data dir')
#     da_hofsj_vx_sel.mean(dim='time').rio.to_raster(os.path.join(path2data_clean, fname_vx))
#     da_hofsj_vy_sel.mean(dim='time').rio.to_raster(os.path.join(path2data_clean, fname_vy))
# else:
#     print(f"File {fname_vx} already exists in cleaned data directory. Skipping save.")


## show average fields (no smoothing)


# # fig,axs=plt.subplots(1,3,figsize=(16,5))
# fig,axs=plt.subplots(1,3,figsize=(16,4))
# da_hofsj_vx_sel.mean(dim='time').plot.imshow(ax=axs[0], vmin=-50, vmax=50, cmap="PiYG")
# axs[0].set_title('vx average (m/yr)')
# da_hofsj_vy_sel.mean(dim='time').plot.imshow(ax=axs[1], vmin=-50, vmax=50, cmap="PiYG")
# axs[1].set_title('vy average (m/yr)')
# da_hofsj_v_sel.mean(dim='time').plot.imshow(ax=axs[2], vmin=0, vmax=100, cmap='viridis')
# axs[2].set_title('v average (m/yr)')
# [ax.set_aspect('equal') for ax in axs]

# fig.savefig('/Users/mizeboud/Documents/Documents_mizeboud/Projects/ContinuIX/WP1_data/figures/' + \
#                 f'Hofsjokull_velocity_selectYears_mean.jpg', dpi=300)
    

## save SELECTED files to 'cleaned' datafolder 
for year in years_select:
    da_vx = da_hofsj_vx_sel.sel(time=year)
    da_vy = da_hofsj_vy_sel.sel(time=year)
    fname_vx = f'hofsjokull_vx_{year}-{year+1}.tif'
    fname_vy = f'hofsjokull_vy_{year}-{year+1}.tif'

    print(fname_vx)
    if not os.path.exists(os.path.join(path2data_clean, fname_vx)):
        print(f'Saving {fname_vx} to cleaned data dir')
        da_vx.rio.to_raster(os.path.join(path2data_clean, fname_vx))
    else:
        print(f"File {fname_vx} already exists in cleaned data directory. Skipping save.")

    if not os.path.exists(os.path.join(path2data_clean, fname_vy)):
        print(f'Saving {fname_vy} to cleaned data dir')
        da_vy.rio.to_raster(os.path.join(path2data_clean, fname_vy))
    else:
        print(f"File {fname_vy} already exists in cleaned data directory. Skipping save.")

    ## stdev and vertical velo
    da_vx_std = ds_hofsj_vx_std['vx'].sel(time=year)
    da_vy_std = ds_hofsj_vy_std['vy'].sel(time=year)
    da_vz = ds_hofsj_vz['vz'].sel(time=year)

    fname_vz = f'hofsjokull_vz_{year}-{year+1}.tif'
    fname_vx_std = f'hofsjokull_vx-std_{year}-{year+1}.tif'
    fname_vy_std = f'hofsjokull_vy-std_{year}-{year+1}.tif'

     ## STDEVS
    if not os.path.exists(os.path.join(path2data_clean, fname_vx_std)):
        print(f'Saving {fname_vx_std} to cleaned data dir')
        da_vx_std.rio.to_raster(os.path.join(path2data_clean, fname_vx_std))
    if not os.path.exists(os.path.join(path2data_clean, fname_vy_std)):
        print(f'Saving {fname_vy_std} to cleaned data dir')
        da_vy_std.rio.to_raster(os.path.join(path2data_clean, fname_vy_std))

    ## VERTICAL
    if year in years_vz:
        fname_vz = f'hofsjokull_v-vertical_{year}-{year+1}.tif'
        if not os.path.exists(os.path.join(path2data_clean, fname_vz)):
            print(f'Saving {fname_vz} to cleaned data dir')
            da_vz.rio.to_raster(os.path.join(path2data_clean, fname_vz))


#%%

''' ##################################
HOMOGENIZED DATA
- resample to target grid resolution
- fill all NaN values with 0, do something else for DEM
- assing all attributes and assemble netcdf

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

target_res = 25 # meter
target_crs = 'EPSG:3057'
da_dummy_target = datafuncs.create_regular_dummy_grid(da_dem_2013, 
                                                      grid_res=target_res, 
                                                      crs=target_crs, unit='m')

print(f"dummy grid res: ", da_dummy_target.rio.resolution())

dx = np.abs(np.diff(da_dummy_target['x'].values))
dy = np.abs(np.diff(da_dummy_target['y'].values))
px = float(np.median(np.concatenate([dx, dy])))
if not (np.allclose(dx, px, rtol=1e-3) and np.allclose(dy, px, rtol=1e-3)):
    print(f"dx: {dx}, dy: {dy}, px: {px}")
if not (np.allclose(dx, target_res, rtol=1e-3) and np.allclose(dy, target_res, rtol=1e-3)):
    print(f"Warning: The dummy grid resolution is not consistent with the target resolution of {target_res} m.\n"\
            f'..Actual dx: {np.unique(dx)} (count per value {np.bincount(np.digitize(dx, np.unique(dx)))}), \n'\
            f'         dy: {np.unique(dy)}, (count per value {np.bincount(np.digitize(dy, np.unique(dy)))}), \n'\
            f'         px: {px}')

else:
    print(f"dx: {dx}, dy: {dy}, px: {px} (consistent with target resolution of {target_res} m)")


#%%
'''## DEM for homogenized: 
- Want to have average DEM of time period for the homogenized dataset. 
- But DEM interval is irregular. (2013, 2020, 2023) 
- So: interpolate (linear interpolation) to have artifical DEMs for every year in 2013-2023. Then average over all years.
'''
# ds_DEM_25m = datafuncs.reproject_match_grid(da_dummy_target.expand_dims('time'), ds_DEM, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
ds_DEM_25m = datafuncs.reproject_match_grid(da_dummy_target.expand_dims('time'), ds_DEM_ortho, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)

## first: fill gaps in DEM with last available value of previous DEMS 
ds_DEM_25m_filled = ds_DEM_25m.ffill(dim='time') # forward fill along time dimension, then take mean
## also do backward fill to fill any remaining NaN values (e.g., if first DEM has NaN values)
ds_DEM_25m_filled = ds_DEM_25m_filled.bfill(dim='time') # backward fill along time dimension, then take mean


### %% Interpolate to have annual DEMs
time_interp = np.arange(2013, 2024) # 2013-2023
ds_DEM_interp = ds_DEM_25m_filled.interp(time=time_interp, method='linear')
# ds_DEM_interp.plot.imshow(col='time', col_wrap=4, cmap='terrain', vmin=0, vmax=2000, cbar_kwargs={'fraction':0.02, 'label':'Elevation (m)'})

## Average DEM
da_DEM_avg_25m0 = ds_DEM_interp.mean(dim='time')

## there's still some very small NaN values in the avg DEM:
## - fill these with linear  interpolation
# Interpolate small gaps
da_DEM_avg_25m = (da_DEM_avg_25m0
                    .interpolate_na(
                        dim="x",
                        method="linear",
                        use_coordinate=False,
                        max_gap=10, # max size of gap, in PX if  use_coord is False, otherwise in M (so should then be multiplied by resolution)
                    )
                    .interpolate_na(
                        dim="y",
                        method="linear",
                        use_coordinate=False, ## do not use coords, as xarray needs Y to be increasing (which its not)
                        max_gap=10,
                    )
) ## looks like all gaps are filled (apart from out of bounds areas)

count_invalid = datafuncs.count_nan_values_in_glacier(da_DEM_avg_25m, gdf_outline_2013)
if count_invalid > 0:
    print(f"Warning: There are still {count_invalid} NaN values in the averaged DEM after interpolation and filling.")



#%%
''' ## BED and THICKNESS for homogenized
- re-calculate thickness using this averaged, filled DEM 
'''
## resample BED from 200 to 25 m
da_bed_25m = datafuncs.reproject_match_grid(da_dummy_target, da_bed, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
da_thickness_25m = da_DEM_avg_25m - da_bed_25m
## remove the extremely small area where thickness is negative 
# ## fill those areas with 0 
da_thickness_25m = da_thickness_25m.where(da_thickness_25m > 0, 0)

fig,ax=plt.subplots(1,1,figsize=(6,5))
cmap_t = plt.get_cmap('Blues')
cmap_t.set_bad(color='red') # set color for NaN values
da_thickness_25m.plot.imshow(ax=ax, cmap=cmap_t, vmin=0, vmax=400, cbar_kwargs={'fraction':0.02, 'label':'Thickness (m)'})
gdf_outline_2013.boundary.plot(ax=ax, color='black')
ax.set_title('Thickness 25m (filled gaps)')


#%%
''' ## dhdt homogenized
- also calculate 'average' dhdt over the period, similar as DEM: 
- use annual-interpolated DEMs to calculate annual dhdt; take average over those 

## I check how well da_dhdt_avg from the annual-DEMs 
#   compares to the overall dhdt computed directly from 2013 and 2023 DEMs 
#   and the max difference is 0.0016 so that is acceptably small.
'''
ds_dhdt_25m = ds_DEM_interp.diff(dim='time')  # from M to m/yr
da_dhdt_avg_25m = ds_dhdt_25m.mean(dim='time') # average over all years

# Interpolate small gaps
max_gap = 5
da_dhdt_avg_25m = (da_dhdt_avg_25m
                    .interpolate_na( dim="x", method="linear", use_coordinate=False, max_gap=max_gap )
                    .interpolate_na( dim="y", method="linear", use_coordinate=False, max_gap=max_gap )
                    )

count_invalid = datafuncs.count_nan_values_in_glacier(da_dhdt_avg_25m, gdf_outline_2013)
if count_invalid > 0:
    print(f"Warning: There are still {count_invalid} NaN values in the da.")


#%%
''' ### Velocity homogenized
- resample each time slice to 25m
- interpolate 2018 to get annual values
- take average
'''
# years_select = [2017, 2019, 2020] ## selected years
years_interp = [2017, 2018, 2019, 2020] ## add 2018 and interpolate to have annual values for 2017-2020
da_hofsj_vx_annual = da_hofsj_vx_sel.interp(time=time_interp, method='linear')
da_hofsj_vy_annual = da_hofsj_vy_sel.interp(time=time_interp, method='linear')
da_hofsj_vx_25m = datafuncs.reproject_match_grid(da_dummy_target.expand_dims('time'), da_hofsj_vx_annual, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
da_hofsj_vy_25m = datafuncs.reproject_match_grid(da_dummy_target.expand_dims('time'), da_hofsj_vy_annual, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
## average over years
da_vx_avg_25m = da_hofsj_vx_25m.mean(dim='time')
da_vy_avg_25m = da_hofsj_vy_25m.mean(dim='time')

## unct: average over years
da_vx_std_25m = ds_hofsj_vx_std['vx'].sel(time=years_select).mean(dim='time')
da_vy_std_25m = ds_hofsj_vy_std['vy'].sel(time=years_select).mean(dim='time')



#%%
'''## ELEVATION BINS: discretize DEM into bins of 50m'''
hmin = da_DEM_avg_25m.min().item()
hmax = da_DEM_avg_25m.max().item()
print('hmin, hmax', np.round(hmin,0), np.round(hmax,0))

da_elev_bins_25m, elev_bin_edges_25m = datafuncs.dicretize_elevation_bins(da_DEM_avg_25m,
                                                     hmin=hmin, hmax=hmax,
                                                     binstep=50)

## save to CLEAN dir
fname = f'hofsjokull_elev-bins.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    print(f'Saving {fname} to cleaned data dir')
    da_elev_bins_25m.rio.to_raster(os.path.join(path2data_clean, fname))


#%%
'''## OUTLINE TO MASK
for hofsjokull: use thickness = 0 as mask, since no outline was provided'''
# burn outline into raster mask
da_outline_mask = (da_dummy_target*2013).rio.clip(gdf_outline_2013.geometry, gdf_outline_2013.crs, drop=False) # drop=False to keep the same grid and not drop the pixels outside the outline (which will be set to nodata)

## hofsjokull: also provide flowbasin mask
da_basin_mask = da_dummy_target.copy(data=np.nan*np.ones_like(da_dummy_target.data)) # initialize with NaN values
for idx, gdf_basin in gdf_basins_2013.iterrows(): # idx starts at 1
    gdf_basin = gpd.GeoDataFrame([gdf_basin], crs=gdf_basins_2013.crs) # convert to GeoDataFrame
    da_mask_i = idx*da_dummy_target.rio.clip(gdf_basin.geometry, gdf_basin.crs, drop=False)
    # da_mask_i.plot.imshow()
    da_basin_mask = da_basin_mask.combine_first(da_mask_i) # combine with previous mask
    ## print basin ID and RGIId
    print(f'Basin {idx} : {gdf_basin.RGIId.values[0]}')


#%%
''' ##################################
Global uncertainties
################################## '''

'''# unct_thickness : unkown '''
'''unct_dhdt : unkown '''
'''unct_velo : stdev''' 

#%%
''' ##################################
RESAMPLING TO TARGET GRID
################################## '''

## initial check that all variables have the same CRS, resolution and shape
da_var_dict = { 'BED':da_bed_25m.copy().rename('BED'),
                'DEM': da_DEM_avg_25m.copy().rename('DEM'),
                'ELEVBINS': da_elev_bins_25m.copy().rename('ELEVBINS'),
                'THK':da_thickness_25m.copy().rename('THK'),
                'DHDT': da_dhdt_avg_25m.copy().rename('DHDT'),
                'VX': da_vx_avg_25m.copy().rename('VX'),
                'VY': da_vy_avg_25m.copy().rename('VY'),
                'ICEMASK': da_outline_mask.copy().rename('ICEMASK'),
                'BASINMASK': da_basin_mask.copy().rename('BASINMASK'),
                'UNCT_VX': da_vx_std_25m.copy().rename('UNCT_VX'),
                'UNCT_VY': da_vy_std_25m.copy().rename('UNCT_VY'),
                
}
## resample to target grid where necessary
for varname , var in da_var_dict.items():
    if var.rio.resolution()[0] != target_res:
        print(f'.. resampling {varname} from {var.rio.resolution()[0]} m to {target_res} m')
        var_target_res = datafuncs.reproject_match_grid(da_dummy_target, var, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)

        ## put back in dictionary
        da_var_dict[varname] = var_target_res
        print(var_target_res.rio.resolution(), var_target_res.shape)
    if var.shape != da_dummy_target.shape:
        print(f'.. reshaping {varname} from {var.shape} m to {da_dummy_target.shape} m')
        var_target_res = datafuncs.reproject_match_grid(da_dummy_target, var, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)

        ## put back in dictionary
        da_var_dict[varname] = var_target_res
        print(var_target_res.rio.resolution(), var_target_res.shape)
        # raise ValueError(f"Shape of {varname} does not match target shape: {var_target_res.shape} vs {da_dummy_target.shape}")
    
assert all(da.rio.crs == da_dummy_target.rio.crs for da in da_var_dict.values()), "Not all variables have the same CRS"
assert all(da.rio.resolution() == da_dummy_target.rio.resolution() for da in da_var_dict.values()), "Not all variables have the same resolution"
assert all(da.shape == da_dummy_target.shape for da in da_var_dict.values()), "Not all variables have the same shape"


print([ da.rio.resolution() for da in da_var_dict.values()])


#%%

'''
# SET ATTRIBUTES OF VARIABLES
Handle NaN values 
'''

da_outline_mask = (da_var_dict['ICEMASK'].copy()
                #    .fillna(0) # fill NaN values with 0 (outside outline)
                   .assign_attrs({'long_name':'Glacier Outline Mask',
                                  'units':'year',
                                  'uncertainty':'unknown',
                                  'crs':target_crs,
                                  'timestamp':'2013',
                                  'description': 'Value is max year of valid glaciated pixel', #; 0 for non-glaciated pixels.',
                                #   'nodata': 0
                                  })
                    .rio.write_crs(target_crs)
)

da_basin_mask = (da_var_dict['BASINMASK'].copy()
                #    .fillna(0) # fill NaN values with 0 (outside outline)
                   .assign_attrs({'long_name':'Basin Mask',
                                  'units':'year',
                                  'uncertainty':'unknown',
                                  'crs':target_crs,
                                  'timestamp':'2013',
                                  'description': 'Value indicates basin label that can be linked to a RGI-v6 glacier ID.',
                                #   'nodata': 0
                                  })
                    .rio.write_crs(target_crs)
)


da_dem_hmg = (da_var_dict['DEM'].copy()
                .assign_attrs({'long_name':'Elevation',
                                'units':'m',
                                  'uncertainty':'unknown',
                                'crs':target_crs,
                                'timestamp':'"2018"',
                                'description':'Average elevation data from DEMs in 2013, 2020 and 2023 (orthometric height).'
                                })
                    .rio.write_crs(target_crs)
    )

da_bedrock_hmg = (da_var_dict['BED'].copy()
                    # .fillna(-999) # fill NaN values with -999
                    .assign_attrs({'long_name':'Bedrock Elevation',
                                   'units':'m',
                                  'uncertainty':'unknown',
                                   'crs':target_crs,
                                   'timestamp':'2013',
                                   'description':'bedrock elevation. Echosounding performed in 1983 but bias-corrected to an additional 2013 survey.'
                                   })
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
                                  'uncertainty':'n/a',
                                  'crs':target_crs,
                                  'timestamp':'"2018"',
                                  'description': f'Discretized elevation values into bins of 50 m. Using lowest (left-edge) value for each bin. Determined from average DEM over 2013-2023.'
                                  })
                    .rio.write_crs(target_crs)
)

## thickness, dhdt, velo: can fill NaN with 0
da_thickness_hmg = (da_var_dict['THK'].copy()
                    .fillna(0)
                    .assign_attrs({'long_name':'Ice Thickness',
                                   'units':'m',
                                  'uncertainty':'unknown',
                                   'crs':target_crs,
                                   'timestamp':'"2018"',
                                   'description':'Ice thickness calculated from DEM-bedrock, using average DEM of 2013-2023. Missing/NaN values were filled with 0.',
                                   'nodata': 0})
                .rio.write_crs(target_crs)
                    )
da_dhdt_hmg = (da_var_dict['DHDT'].copy()
               .fillna(0)
               .assign_attrs({'long_name':'Surface Elevation Change',
                              'units':'m/year',
                                  'uncertainty':'unknown',
                              'crs':target_crs,
                              'timestamp':'"2013-2023"',
                              'description':'Average annual elevation change, obtained from DEMs in 2013, 2020 and 2023. Missing/NaN values were filled with 0.',
                              'nodata': 0})
                .rio.write_crs(target_crs)
               )

da_vx_hmg = (da_var_dict['VX'].copy()
             .fillna(0)
             .assign_attrs({'long_name': 'surface ice velocity (x-component)',
                            'units':'m/year',
                                  'uncertainty':'provided as grid, UNCT_VX',
                            'crs':target_crs,
                            'timestamp':'2017-2020',
                            'description':'Average velocity for the period 2017-2020, calculated from observations in 2017, 2019, 2020. Missing/NaN values were filled with 0.',
                            'nodata': 0
                            })
                .rio.write_crs(target_crs)
)

da_vy_hmg = (da_var_dict['VY'].copy()
             .fillna(0)
             # .rename('VY')
             .assign_attrs({'long_name': 'surface ice velocity (y-component)',
                            'units':'m/year',
                            'uncertainty':'provided as grid, UNCT_VY',
                            'crs':target_crs,
                            'timestamp':'2017-2020',
                            'description':'Average velocity for the period 2017-2020, calculated from observations in 2017, 2019, 2020. Missing/NaN values were filled with 0.',
                            'nodata': 0
                            })
                .rio.write_crs(target_crs)
)

da_vx_std_hmg = (da_var_dict['UNCT_VX'].copy()
             .fillna(0)
             .assign_attrs({'long_name': 'surface ice velocity uncertainty (x-component)',
                            'units':'m/year',
                            'uncertainty':'n/a',
                            'crs':target_crs,
                            'timestamp':'2017-2020',
                            'description':'Standard deviation of velocity component, averaged for the period 2017-2020. Missing/NaN values were filled with 0.',
                            'nodata': 0
                            })
                .rio.write_crs(target_crs)
)

da_vy_std_hmg = (da_var_dict['UNCT_VY'].copy()
             .fillna(0)
             .assign_attrs({'long_name': 'surface ice velocity uncertainty (y-component)',
                            'units':'m/year',
                            'uncertainty':'n/a',
                            'crs':target_crs,
                            'timestamp':'2017-2020',
                            'description':'Standard deviation of velocity component, averaged for the period 2017-2020. Missing/NaN values were filled with 0.',
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
                da_basin_mask,
                da_vx_std_hmg,
                da_vy_std_hmg,
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

# '''# check values by plotting
# '''
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

fname_nc = 'hofsjokull_glacier_observations.nc'

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

with xr.open_dataset(
        os.path.join(path2data_homog, fname_nc),
        decode_coords="all" # decode_coords="all" is important when reopening NetCDFs with rioxarray-style CRS metadata; otherwise the CRS may appear to be missing.
    ) as ds_glacier_loaded:
    
    print('CRS:', ds_glacier_loaded.rio.crs)
    print('spatial_ref attrs:', ds_glacier_loaded["spatial_ref"].attrs)
    assert ds_glacier_loaded.rio.crs is not None, "CRS is missing in the loaded dataset"

#%%
#
fname_nc = 'hofsjokull_glacier_observations.nc'

with xr.open_dataset(
        os.path.join(path2data_homog, fname_nc),
        decode_coords="all" # decode_coords="all" is important when reopening NetCDFs with rioxarray-style CRS metadata; otherwise the CRS may appear to be missing.
    ) as ds_glacier_loaded:
    
    print('CRS:', ds_glacier_loaded.rio.crs)
    print('spatial_ref attrs:', ds_glacier_loaded["spatial_ref"].attrs)
    print('resolution:', ds_glacier_loaded.rio.resolution())
    assert ds_glacier_loaded.rio.crs is not None, "CRS is missing in the loaded dataset"

## check values by plotting
fig,axs=plt.subplots(2,4, figsize=(16,8))
row,col = 0,0
for var, cmap, vminmax in zip(  ['BED',     'DEM',     'ELEVBINS', 'THK', 'DHDT', 'VX', 'VY',   'ICEMASK'],
                                ['cividis','cividis','cividis',  'Blues', 'RdBu','PiYG','PiYG',  'viridis'],
                                [(0.6e3, 1.8e3), (0.6e3, 1.8e3), (0.6e3, 1.8e3), (0,700),  (-5,5), (-100,100),(-100,100),None]):
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


# fig.savefig(os.path.join(path2data_homog, 'hofsjokull_netcdf_vars.png'), dpi=300)
