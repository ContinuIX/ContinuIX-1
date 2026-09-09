#%% 
## 
# M. Izeboud, June 2026
''' Check and pre-processing of synthetic glaciers
Synthetic data has been submitted by data contributor XXX.

The data has been checked and cleaned.
--> no actual data processing has occured for the synthetic glaciers, just making sure everything is in the same format as the other glaciers (attributes; metadata, variable names, etc.)

# M. Izeboud, June 2026
'''
import numpy as np
import geopandas as gpd
import xarray as xr
import os
import matplotlib.pyplot as plt
import rasterio as rio
import rioxarray #  activates .rio accessor of xarray
import warnings

import datafunctions as datafuncs

#%% 
''' ########################
Synth - ICEPACK
###########################
'''

path2data_raw = '../../ContinuIX_WP1_data/Data_Package/01_submitted_data/Synthetic-Icepack/'
path2data_clean = '../../ContinuIX_WP1_data/Data_Package/02_raw-cleaned_data/Synthetic-Icepack/'
path2data_homog = '../../ContinuIX_WP1_data/Data_Package/03_homogenized_data/Synthetic-Icepack/'


ds_unperturbed = xr.open_dataset(os.path.join(path2data_raw, 'unperturbed.nc'))
ds_unperturbed

var_to_save_hmg = ['bed','surface','thickness','velocity_x','velocity_y','dhdt','thickness_unc','dhdt_unc'] 
new_var_names_mhg = ['BED','DEM',   'THK',      'VX',           'VY',   'DHDT',  'UNCT_THK',   'UNCT_DHDT']


# target_crs = 'n/a'
target_crs = 'EPSG:3857' # Web Mercator (meters); doesnt really matter for synthetic though

ds_synth_hmg = ds_unperturbed[var_to_save_hmg].copy().rename({old_var: new_var for old_var, new_var in zip(var_to_save_hmg, new_var_names_mhg)})
ds_synth_hmg['UNCT_VX'] = ds_unperturbed['velocity_unc']
ds_synth_hmg['UNCT_VY'] = ds_unperturbed['velocity_unc']

## re-make elevation bins

## multiple DEMs: use first to do the binning, but use the min-max range of both to define bin range
# hmin = np.min([da_dem17.min().item(), da_dem23.min().item()]) 
# hmax = np.max([da_dem17.max().item(), da_dem23.max().item()]) 
hmin = ds_synth_hmg['DEM'].min().item()
hmax = ds_synth_hmg['DEM'].max().item()
da_elev_bins, elev_bin_edges = datafuncs.dicretize_elevation_bins(ds_synth_hmg['DEM'], 
                                                     hmin=hmin, hmax=hmax,
                                                     binstep=50)
da_elev_bins

ds_synth_hmg['ELEVBINS'] = da_elev_bins
ds_synth_hmg['ICEMASK'] = xr.where(ds_synth_hmg['BED'] > 0, 1, 0) # ice mask: 1 for ice, 0 for no ice


''' ## Make grid SQUARE (is 1.2 by 1)'''
ds = ds_synth_hmg.copy()
grid_res = 1.0 ; unit='m' 
target_res = 1

x0 = ds.x.min().item() ; x1 = ds.x.max().item() ; y0 = ds.y.min().item() ; y1 = ds.y.max().item()
## make grid coordinates start and end at multiples of grid_res to avoid floating point precision issues
x0 = np.floor(x0/grid_res)*grid_res; x1 = np.floor(x1/grid_res)*grid_res; 
y0 = np.floor(y0/grid_res)*grid_res; y1 = np.floor(y1/grid_res)*grid_res
x_seq = np.arange(x0, x1+grid_res, step=grid_res )
y_seq = np.arange(y0, y1+grid_res, step=grid_res )
dy = np.unique(ds.y.diff(dim='y').values)
## check if y_seq is decreasing and reverse if needed
if dy < 0: # if y resolution is negative, then y_seq should be decreasing
    y_seq = y_seq[::-1]

### set up dummy grid
da_dummy_target = xr.DataArray(
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
)
## interpolate data onto grid
ds_synth_hmg = ds_synth_hmg.interp(x=da_dummy_target.x, y=da_dummy_target.y, method='linear')
## update attribute of x
ds_synth_hmg.attrs['dx_m'] = 1.0
## write CRS to all variables
ds_synth_hmg = (ds_synth_hmg.rio.set_spatial_dims(x_dim="x", y_dim="y") # Make sure spatial dims are known
                    # Write CRS and CF grid mapping to the whole dataset
                    .rio.write_crs(target_crs)
                    .rio.write_grid_mapping("spatial_ref")
                    .rio.write_transform()
)
for var in ds_synth_hmg.data_vars:
    if var != "spatial_ref":
        da = ds_synth_hmg[var]
        da = da.rio.write_crs(target_crs, inplace=True)
        ds_synth_hmg[var] = da
        ds_synth_hmg[var].attrs["grid_mapping"] = "spatial_ref"

dx = np.abs(np.diff(ds_synth_hmg['x'].values))
dy = np.abs(np.diff(ds_synth_hmg['y'].values))
px = float(np.median(np.concatenate([dx, dy])))
if not (np.allclose(dx, px, rtol=1e-3) and np.allclose(dy, px, rtol=1e-3)):
    print(f"dx: {dx}, dy: {dy}, px: {px}")
if not (np.allclose(dx, target_res, rtol=1e-3) and np.allclose(dy, target_res, rtol=1e-3)):
    print(f"Warning: The dummy grid resolution is not consistent with the target resolution of {target_res} m.\n"\
            f'..Actual dx: {np.unique(dx)} (count per value {np.bincount(np.digitize(dx, np.unique(dx)))}), \n'\
            f'         dy: {np.unique(dy)}, (count per value {np.bincount(np.digitize(dy, np.unique(dy)))}), \n'\
            f'         px: {px}')

# ds_perturbed = xr.open_dataset(os.path.join(path2data_raw, 'perturbed.nc'))
# ds_perturbed

## encoding settings for compression and data type; same for all variables
comp = {"zlib": True, 
        "complevel": 5,  ## level of compression; higher number = more compression but slower read/write
        "dtype": "float32", ## 7 digits of precision 
        }
encoding = {var: comp for var in ds_synth_hmg.data_vars if var != "spatial_ref"}  # Exclude spatial_ref from encoding
encoding["spatial_ref"] = {}  # No compression for spatial_ref

fname_nc = 'synthetic-icepack_glacier.nc'

try:
    print('--> saving homogenized data to netcdf; overwriting if file exists')
    ds_synth_hmg.to_netcdf(os.path.join(path2data_homog, fname_nc), 
                            mode='w', format='NETCDF4', 
                            engine='netcdf4',
                            encoding=encoding 
    )
    ds_synth_hmg.close()

except PermissionError:
    print('--> CHECK INPUT WINDOW')
    answ = input(f"PermissionError to write {fname_nc}. Input Y to overwrite")
    if answ == 'Y' or answ == 'y':
        print('..removing existing and re-saving file')
        os.remove(os.path.join(path2data_homog, fname_nc))
        ds_synth_hmg.to_netcdf(os.path.join(path2data_homog, fname_nc), 
                            mode='w', format='NETCDF4', 
                            engine='netcdf4',
                            encoding=encoding 
        )
        ds_synth_hmg.close()
    else: print('..aborted saving file')



with xr.open_dataset(
        os.path.join(path2data_homog, fname_nc),
        decode_coords="all" # decode_coords="all" is important when reopening NetCDFs with rioxarray-style CRS metadata; otherwise the CRS may appear to be missing.
    ) as ds_glacier_loaded:
    
    print('resolution:', ds_glacier_loaded.rio.resolution())
    print('CRS:', ds_glacier_loaded.rio.crs)

## check values by plotting
fig,axs=plt.subplots(2,4, figsize=(16,5))
row,col = 0,0
for var, cmap, vminmax in zip(  ['BED',     'DEM',          'ELEVBINS',          'THK', 'DHDT', 'VX', 'VY',   'ICEMASK'],
                                ['cividis','cividis',       'cividis',          'Blues', 'RdBu','PiYG','PiYG',  'viridis'],
                                [(0.6e3, 1.8e3), (0.6e3, 1.8e3), (0.6e3, 1.8e3), (0,700),  (-2e-7,2e-7), (-100,100),(-100,100),None]):
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
# [ax.set_axis_off() for ax in axs.flatten()];

fig.savefig(os.path.join(path2data_homog, 'synthetic-icepack_netcdf_vars.png'), dpi=300)


# %%
''' ########################
Synth - ALETSCH
###########################
'''

path2data_raw = '../../ContinuIX_WP1_data/Data_Package/01_submitted_data/Synthetic-Aletsch/'
path2data_clean = '../../ContinuIX_WP1_data/Data_Package/02_raw-cleaned_data/Synthetic-Aletsch/'
path2data_homog = '../../ContinuIX_WP1_data/Data_Package/03_homogenized_data/Synthetic-Aletsch/'

target_crs = 'EPSG:2056'
target_res = 100
ds_aletsch_raw_subm = xr.open_dataset(os.path.join(path2data_raw, 'syn-aletsch_1sigma_v01_raw.nc'))
ds_aletsch_ref = xr.open_dataset(os.path.join(path2data_raw, 'syn-aletsch_1sigma_v01_ref.nc'))


var_to_save = ['BED','DEM','THK','VX','VY','DHDT','mask','UNCT_THK','UNCT_DHDT','UNCT_VX','UNCT_VY'] 

ds_aletsch_raw = ds_aletsch_raw_subm[var_to_save].copy()
ds_aletsch_hmg = ds_aletsch_ref[var_to_save].copy()

## create ELEVBINS
hmin = ds_aletsch_raw['DEM'].min().item()
hmax = ds_aletsch_raw['DEM'].max().item()
da_elev_bins_raw, elev_bin_edges = datafuncs.dicretize_elevation_bins(ds_aletsch_raw['DEM'], 
                                                     hmin=hmin, hmax=hmax,
                                                     binstep=50)
da_elev_bins_hmg, elev_bin_edges = datafuncs.dicretize_elevation_bins(ds_aletsch_ref['DEM'],
                                                     hmin=hmin, hmax=hmax,
                                                        binstep=50)

ds_aletsch_raw['ELEVBINS'] = da_elev_bins_raw
ds_aletsch_hmg['ELEVBINS'] = da_elev_bins_hmg

## update CRS and attributes
for var in ds_aletsch_raw.data_vars:
    if var != "spatial_ref":
        da_raw = ds_aletsch_raw[var]
        da_raw = da_raw.rio.write_crs(target_crs, inplace=True)
        da_hmg = ds_aletsch_hmg[var]
        da_hmg = da_hmg.rio.write_crs(target_crs, inplace=True)
        if var == 'mask':
            ds_aletsch_raw['ICEMASK'] = da_raw
            ds_aletsch_raw['ICEMASK'].attrs["grid_mapping"] = "spatial_ref"
            ds_aletsch_raw = ds_aletsch_raw.drop_vars('mask')
            ds_aletsch_hmg['ICEMASK'] = da_hmg
            ds_aletsch_hmg['ICEMASK'].attrs["grid_mapping"] = "spatial_ref"
            ds_aletsch_hmg = ds_aletsch_hmg.drop_vars('mask')
        else:
            ds_aletsch_raw[var] = da_raw
            ds_aletsch_raw[var].attrs["grid_mapping"] = "spatial_ref"
            ds_aletsch_hmg[var] = da_hmg
            ds_aletsch_hmg[var].attrs["grid_mapping"] = "spatial_ref"

## update some attributes
attrs_to_update = {'grid_resolution': '100 m', 
                   'timestamp': 'n/a', 
                   'title': 'Synthetic data for an idealized Aletsch glacier'}
for attr, value in attrs_to_update.items():
    ds_aletsch_raw.attrs[attr] = value
    ds_aletsch_hmg.attrs[attr] = value


''' ### save to NC '''
## encoding settings for compression and data type; same for all variables
comp = {"zlib": True, 
        "complevel": 5,  ## level of compression; higher number = more compression but slower read/write
        "dtype": "float32", ## 7 digits of precision 
        }

### save RAW
encoding = {var: comp for var in ds_aletsch_raw.data_vars if var != "spatial_ref"}  # Exclude spatial_ref from encoding
encoding["spatial_ref"] = {}  # No compression for spatial_ref

fname_nc = 'synthetic-aletsch_glacier.nc'
if not os.path.exists(os.path.join(path2data_clean, fname_nc)):
    ds_aletsch_raw.to_netcdf(os.path.join(path2data_clean, fname_nc), 
                            mode='w', format='NETCDF4', 
                            engine='netcdf4',
                            encoding=encoding 
    )
    ds_aletsch_raw.close()

### save HMG
fname_nc = 'synthetic-aletsch_glacier.nc'

encoding = {var: comp for var in ds_aletsch_raw.data_vars if var != "spatial_ref"}  # Exclude spatial_ref from encoding
encoding["spatial_ref"] = {}  # No compression for spatial_ref

if not os.path.exists(os.path.join(path2data_homog, fname_nc)):
    ds_aletsch_hmg.to_netcdf(os.path.join(path2data_homog, fname_nc), 
                            mode='w', format='NETCDF4', 
                            engine='netcdf4',
                            encoding=encoding 
    )
    ds_aletsch_hmg.close()

## plot 

with xr.open_dataset(
        os.path.join(path2data_clean, fname_nc),
        # os.path.join(path2data_homog, fname_nc),
        decode_coords="all" # decode_coords="all" is important when reopening NetCDFs with rioxarray-style CRS metadata; otherwise the CRS may appear to be missing.
    ) as ds_glacier_loaded:
    
    print('resolution:', ds_glacier_loaded.rio.resolution())
    print('CRS:', ds_glacier_loaded.rio.crs)

## check values by plotting
fig,axs=plt.subplots(2,4, figsize=(16,5))
row,col = 0,0
for var, cmap, vminmax in zip(  ['BED',     'DEM',          'ELEVBINS',          'THK', 'DHDT', 'VX', 'VY',   'ICEMASK'],
                                ['cividis','cividis',       'cividis',          'Blues', 'RdBu','PiYG','PiYG',  'viridis'],
                                [(1500, 3000), (1500, 3000), (1500, 3000), (0,700),  (-5,5), (-100,100),(-100,100),None]):
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

fig.savefig(os.path.join(path2data_clean, 'synthetic-aletsch_netcdf.png'), dpi=300)
# fig.savefig(os.path.join(path2data_homog, 'synthetic-aletsch_netcdf.png'), dpi=300)


# %%
