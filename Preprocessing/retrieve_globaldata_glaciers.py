#%%
''' Retrieve data for all glaciers from globally-available datasets
- Hugonnet et al. (2021) for elevation change (DHDT)
- Millan et al. (2022) for velocity (VX, VY) and thickness (THK)
- Farinotti et al. (2019) for thickness (THK)
- Maffezzoli et al. (2023) for thickness (THK)

All data has been downloaded from public repositories. The original folder structures are used to load the data.
Data is extracted for the respective studied glaciers, by loading the relevant files, combining them and clipping to the same glacier domain used for the observational datasets.

# M. Izeboud, July 2026

'''

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import os
import matplotlib.pyplot as plt
import rasterio as rio
import rioxarray #  activates .rio accessor of xarray
from shapely.geometry import Polygon
import warnings
import json
import datafunctions as datafuncs

path2data_input = '../../ContinuIX_WP1_data/Global_data/'
path2data_homog = '../../ContinuIX_WP1_data/Data_Package/03_homogenized_data/'
path2data_global = '../../ContinuIX_WP1_data/Data_Package/04_global_data/'

glacier_dirnames = ['Aletsch', 'Argentiere', 'Gepatschferner', 'Hofsjokull', 'SaryTor', 'Zongo']
glacier_names = [name.lower() for name in glacier_dirnames] 
glacier_RGIregions = dict(zip(glacier_dirnames, ['11','11','11','06','13','16']))

#%%

# calculate image boundaries
def img_bound_gpd(img_da): # input: xarray dataArray
    '''Create shapely polygon based on boundaries of xr.DataArray'''
    polygon_geom = Polygon.from_bounds(*img_da.rio.bounds())
    polygon = gpd.GeoDataFrame(index=[0], crs=img_da.rio.crs, geometry=[polygon_geom])  
    return polygon


def load_combine_tilelist_in_domain(filelist_tiles, da_to_match, resample_method=rio.enums.Resampling.nearest):
    '''Load and combine a list of raster tiles that intersect with the domain of a given (empty) reference xarray dataArray (i.e. stitching the tiles together). The tiles are reprojected to match the grid of the reference dataArray. 
    The function returns the combined dataArray and the sub-list of tile-files that intersect with the reference domain.
    Used for both Hugonnet and Millan datasets, which are provided as multiple tiles.

    Parameters:
        filelist_tiles (list)           : List of file paths to raster tiles.
        da_to_match (xarray.DataArray)  : Reference dataArray whose domain is used to filter tiles
        resample_method (rasterio.enums.Resampling): Resampling method for reprojecting tiles.
    Returns:
        da_tiles (xarray.DataArray) : Combined dataArray of all intersecting tiles, reprojected to match the reference grid.
        tilelist_intersect (list)   : List of file paths of tiles that intersect with the reference domain.
    '''
    import rasterio as rio
    # import warnings

    ## load tiles intersechting with reference dataArray domain (da_to_match)
    da_datatiles_list = []
    tilelist_intersect =[]
    for tile_file in filelist_tiles:
        with xr.open_dataarray(tile_file).isel(band=0).drop_vars('band') as da_tile:
            ''' make sure file has valid CRS'''
            if not da_tile.rio.crs:
                raise ValueError(f"CRS is missing for {tile_file}. Cannot reproject without a valid CRS.")
            # ''' check if tile intersects domain'''
            # tile_bound_gpd   = img_bound_gpd(da_tile)
            # target_bound_gpd = img_bound_gpd(da_to_match)
            # if tile_bound_gpd.to_crs(target_bound_gpd.crs).intersects(target_bound_gpd).values:
            
            '''## get tile and domain bounds'''
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning) ## ignore 'rectified to skew grid conversion' warnings 
                target_bound_gpd = img_bound_gpd(da_to_match)
                tile_bound_gpd   = img_bound_gpd(da_tile).to_crs(target_bound_gpd.crs)
            
            '''## first check: if tile is less then 1 km2, it's definitely too small for our glacier domains [this is relevant for Maffezzoli thickness data]'''
            if tile_bound_gpd['geometry'].area.values < 1e6: # m2
                continue

            ''' check if tile intersects domain'''
            if tile_bound_gpd.intersects(target_bound_gpd).values:
                # print('tile intersects bounds of dataArray: ', os.path.basename(tile_file))
                # ''' reproject hugo data to desired grid ''' --> do reprojections at later stage? No do here, since sometimes I load huge regional files and want to have only glacier (for speed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning) ## ignore 'rectified to skew grid conversion' warnings
                    da_tile_reprj = datafuncs.reproject_match_grid(da_to_match, da_tile, resample_method=resample_method)
                da_datatiles_list.append(da_tile_reprj)
                # da_datatiles_list.append(da_tile)
                tilelist_intersect.append(tile_file)

    '''## combine to dataset
    If data tiles have gaps with nodata values (which occurs in Hugonnet dataset), that's an issue with the automatic xarray combining/reprojecting; giving strange artifacts
    Instead, each datatile needs to be reprojected first to the whole domain (just filling with empty nans) and then the stack of all the same domain can be combined.
    '''

    da_tiles = da_datatiles_list[0] # initialise domain wide dataset contianing 1 region of data
    for ds in da_datatiles_list[1:]:
        # combine_first() defaults to non-null values in the calling object, and fills holes with called object.
        # effecitvely patching all regions to the first
        da_tiles = da_tiles.combine_first(ds) 
    
    return da_tiles, tilelist_intersect



def save_encoded_nc(ds_glacier, path2data, fname_nc, comp=None):
    ''' Save xarray dataset to netcdf with encoding settings for compression and data type. If the file already exists, it will be overwritten.'''

    # Force each data variable to point to spatial_ref (needed to save properly with CRS info)
    for var in ds_glacier.data_vars:
        if var != "spatial_ref":
            ds_glacier[var].attrs["grid_mapping"] = "spatial_ref"


    ## encoding settings for compression and data type; same for all variables
    if comp is None:
        comp = {"zlib": True, 
                "complevel": 5,     ## level of compression; higher number = more compression but slower read/write
                "dtype": "float32", ## 7 digits of precision 
                # "_FillValue": np.float32(-999), ## for remaining NaN values
                }
        encoding = {var: comp for var in ds_glacier.data_vars if var != "spatial_ref"}  # Exclude spatial_ref from encoding
        encoding["spatial_ref"] = {}  # No compression for spatial_ref

    try:
        print('--> saving global data to netcdf; overwriting if file exists')
        ds_glacier.to_netcdf(os.path.join(path2data, fname_nc), 
                                mode='w', format='NETCDF4', 
                                engine='netcdf4',
                                encoding=encoding  ## use encoding to compress file size. 
        )
        ds_glacier.close()

    except PermissionError:
        # print('--> CHECK INPUT WINDOW')
        # answ = input(f"PermissionError to write {fname_nc}. Input Y to overwrite")
        # if answ == 'Y' or answ == 'y':
        print('!! removing existing and re-saving file')
        os.remove(os.path.join(path2data, fname_nc))
        ds_glacier.to_netcdf(os.path.join(path2data, fname_nc), 
                            mode='w', format='NETCDF4', 
                            engine='netcdf4',
                            encoding=encoding 
        )
        ds_glacier.close()
        # else: print('..aborted saving file')


#%% Aletsch
import time 
## load homogenized data to get reference grid and ref
glacier_Name = 'SaryTor'
glacier_Name = 'Aletsch'

for glacier_Name in glacier_dirnames:
    print(f'-------- {glacier_Name} --------')

    fname_nc_global = f'{glacier_Name.lower()}_globaldata*.nc'
    if len([f for f in os.listdir(path2data_global) if f.startswith(f'{glacier_Name.lower()}_globaldata')]) == 5:
        print(f'.. All global data netcdfs already exists for {glacier_Name}. Skipping.')
        continue

    # if os.path.exists(os.path.join(path2data_global, fname_nc_global)):
    #     print(f'..global data netcdf already exists for {glacier_Name}. Skipping.')
    #     continue

    fname_nc_hmg = f'{glacier_Name}_glacier_observations.nc'
    ds_glacier_homog = xr.open_mfdataset(os.path.join(path2data_homog, glacier_Name, fname_nc_hmg),
                                        decode_coords="all" # decode_coords="all" is important when reopening NetCDFs with rioxarray-style CRS metadata; otherwise the CRS may appear to be missing.
                                        )
    ds_glacier_homog
    glacier_crs = ds_glacier_homog.rio.crs    ## CRS.from_wkt() format
    glacier_crs_str = glacier_crs.to_string() ## 'EPSG:XXXX' format
    glacier_bounds = ds_glacier_homog.rio.bounds()

    ## set up reference grid
    da_grid_glacier = ds_glacier_homog['DEM'].copy(data=np.nan*np.ones_like(ds_glacier_homog['DEM'].data)) # create empty grid to use as reference
    
    # ## reference grid should use resolution of global datasets (not of homogenized datasets)
    res_hugo = 100 # m 
    da_grid_hugo = datafuncs.create_regular_dummy_grid(da_grid_glacier, grid_res=res_hugo, crs=glacier_crs, unit='m', add_buffer=None)
    
    res_mill = 50 # m : both for velo and thickness
    da_grid_mill = datafuncs.create_regular_dummy_grid(da_grid_glacier, grid_res=res_mill, crs=glacier_crs, unit='m', add_buffer=None)

    res_maff = 100 # m
    da_grid_maff = datafuncs.create_regular_dummy_grid(da_grid_glacier, grid_res=res_maff, crs=glacier_crs, unit='m', add_buffer=None)

    ## load filelist
    with open(os.path.join(path2data_input, f'filelist_globalSets.json'), 'r') as f:
        dict_filelist = json.load(f)
    filelist_hugo = dict_filelist['Hugonnet']
    filelist_mill = dict_filelist['Millan']
    filelist_maff = dict_filelist['Maffezzoli']
    filelist_fari = dict_filelist['Farinotti']

    
    ''' ----------------------
    ## open Hugonnet Elevation Change
    The nomenclature of the tiles denotes the south west corner (e.g., N02W040 = 02°-03° latitude; -40°-39° longitude, and S40E129 = -40°-39° latitude; 129°-130° longitude).
    -------------------------- '''
    print('.. loading DHDT hugonnet')
    
    ## get sub-directory for corret RGI region of current glacier
    rgi_dir_options = os.listdir(os.path.join(path2data_input, 'Hugonnet2021'))
    rgi_dir_hugo = [dirname for dirname in rgi_dir_options if glacier_RGIregions[glacier_Name] in dirname] ## need to be vague because sometimes RGI region number is specific in dirname, sometimes its part of a group
    assert len(rgi_dir_hugo) == 1
    path2hugo = os.path.join(path2data_input, 'Hugonnet2021', rgi_dir_hugo[0], 'dhdt')
    dhdt_list = sorted(os.listdir(path2hugo))
    dhdt_list = [f for f in dhdt_list if f.endswith('.tif')] ## only tif files
    dhdt_list = [os.path.join(path2hugo, f) for f in dhdt_list] ## full pathnames
    dhdt_err_list = [f.replace('dhdt','dhdt_err') for f in dhdt_list]


    ## load files into a single dataArray
    da_dhdt, tilelist_domain_hugo = load_combine_tilelist_in_domain(dhdt_list, da_grid_hugo, 
                                            resample_method=rio.enums.Resampling.nearest
                                            )
    da_dhdt_err, _ = load_combine_tilelist_in_domain(dhdt_err_list, da_grid_hugo, 
                                            resample_method=rio.enums.Resampling.nearest
                                            )
    ## Reproject to glacier CRS and regularized grid for Hugo resolution


    assert da_dhdt.rio.crs == da_dhdt_err.rio.crs == glacier_crs, 'CRS of Hugonnet data does not match CRS of glacier data'
    print('Hugo data RES:', da_dhdt.rio.resolution(), 'Hugo data SHAPE:', da_dhdt.shape)

    
    ## assign attributes
    da_dhdt = da_dhdt.rename('DHDT').fillna(0).assign_attrs({'long_name':'Elevation Change',
                                    'units':'m i.e./yr',
                                    'resolution': f'{res_hugo} m',
                                    'crs':glacier_crs_str,
                                    'timestamp':'2000-2020',
                                    'description':'Annual elevation change from Hugonnet et al. (2021). Nodata areas were filled with 0.',
                                    'nodata':0
                                    })
    da_dhdt_err = da_dhdt_err.rename('UNCT_DHDT').fillna(0).assign_attrs(
                                   {'long_name':'Elevation Change Uncertainty',
                                    'units':'m i.e./yr',
                                    'resolution': f'{res_hugo} m',
                                    'crs':glacier_crs_str,
                                    'timestamp':'2000-2020',
                                    'description':'Annual elevation change uncertainty (1 sigma) from Hugonnet et al. (2021). Nodata areas were filled with 0.',
                                    })
    
    # fig,ax=plt.subplots(figsize=(6,5))
    # da_dhdt.plot.imshow(ax=ax, vmin=-5, vmax=5, cmap='RdBu', cbar_kwargs={'shrink': 0.7})
    # ax.set_title(f'dhdt {glacier_Name}')
    # ax.set_aspect('equal')


    ''' --- ## Save dataset to netcdf ---------------- '''
    
    ds_glacier_nc = (xr.Dataset({   'DHDT': da_dhdt,
                                    'UNCT_DHDT': da_dhdt_err
                    }).assign_attrs({'title':'Glacier observation data of DHDT from Global Dataset',
                        'grid_resolution': f'{str(da_dhdt.rio.resolution())} m',
                        'description':'see attributes of each variable',
                        'timestamp':'',
                    })
                    .rio.set_spatial_dims(x_dim="x", y_dim="y") # Make sure spatial dims are known
                    # Write CRS and CF grid mapping to the whole dataset
                    .rio.write_crs(glacier_crs)
                    .rio.write_grid_mapping("spatial_ref")
                    .rio.write_transform()
    )
    
    fname_nc = f'{glacier_Name.lower()}_globaldata_DHDT-hugonnet.nc'
    save_encoded_nc(ds_glacier_nc, path2data_global, fname_nc, comp=None)

    
    ''' ----------------------
    ## Millan velocity
    -------------------------- '''
    ## get sub-directory for corret RGI region of current glacier
    rgi_dir_options = os.listdir(os.path.join(path2data_input, 'Millan2022/velocity/'))
    rgi_dir_millan = [dirname for dirname in rgi_dir_options if f'RGI-{glacier_RGIregions[glacier_Name].lstrip("0")}' in dirname] ## need to be vague because sometimes RGI region number is specific in dirname, sometimes its part of a group
    assert len(rgi_dir_millan) == 1, "Number of Millan sub-directories not 1"
    path2millan = os.path.join(path2data_input, 'Millan2022/velocity/', rgi_dir_millan[0])
    vx_list = sorted([f for f in os.listdir(path2millan) if 'VX' in f])
    vy_list = sorted([f for f in os.listdir(path2millan) if 'VY' in f])
    stdx_list = sorted([f for f in os.listdir(path2millan) if 'STDX' in f])
    stdy_list = sorted([f for f in os.listdir(path2millan) if 'STDY' in f])
    ## make full path name
    vx_list = [os.path.join(path2millan, f) for f in vx_list] ## full pathnames
    vy_list = [os.path.join(path2millan, f) for f in vy_list] ## full pathnames
    stdx_list = [os.path.join(path2millan, f) for f in stdx_list] ## full pathnames
    stdy_list = [os.path.join(path2millan, f) for f in stdy_list] ## full pathnames

    ## load files into a single dataArray; reproject and match grid of glacier data
    if len(vx_list) > 1: ## for some RGI regions there's multiple files, for other regions theres only 1 file.
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", category=UserWarning) ## ignore 'rectified to skew grid conversion' warnings
           
        da_vx, tilelist_domain_millan = load_combine_tilelist_in_domain(vx_list, da_grid_glacier, resample_method=rio.enums.Resampling.nearest)
        da_vy,   _ = load_combine_tilelist_in_domain(vy_list, da_grid_glacier, resample_method=rio.enums.Resampling.nearest)
        da_stdx, _ = load_combine_tilelist_in_domain(stdx_list, da_grid_glacier, resample_method=rio.enums.Resampling.nearest)
        da_stdy, _ = load_combine_tilelist_in_domain(stdy_list, da_grid_glacier, resample_method=rio.enums.Resampling.nearest)

    elif len(vx_list) == 1:
        da_vx = xr.open_dataarray(vx_list[0]).isel(band=0).drop_vars('band') 
        da_vy = xr.open_dataarray(vy_list[0]).isel(band=0).drop_vars('band')
        da_stdx = xr.open_dataarray(stdx_list[0]).isel(band=0).drop_vars('band')
        da_stdy = xr.open_dataarray(stdy_list[0]).isel(band=0).drop_vars('band')
        tilelist_domain_millan = vx_list.copy()
        ## reproject match
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning) ## ignore 'rectified to skew grid conversion' warnings  
            da_vx = datafuncs.reproject_match_grid(da_grid_glacier, da_vx, resample_method=rio.enums.Resampling.nearest)
            da_vy = datafuncs.reproject_match_grid(da_grid_glacier, da_vy, resample_method=rio.enums.Resampling.nearest)
            da_stdx = datafuncs.reproject_match_grid(da_grid_glacier, da_stdx, resample_method=rio.enums.Resampling.nearest)
            da_stdy = datafuncs.reproject_match_grid(da_grid_glacier, da_stdy, resample_method=rio.enums.Resampling.nearest)    
    else:
        raise ValueError(f"Did not find any files for {glacier_Name} in Millan2022 dataset.")
    
    assert da_vx.rio.crs == da_vy.rio.crs == da_stdx.rio.crs == da_stdy.rio.crs == glacier_crs, 'CRS of Millan data does not match CRS of glacier data'

    ## assign attributes
    da_vx = da_vx.rename('VX').assign_attrs({'long_name': 'surface ice velocity (x-component)',
                            'units':'m/year',
                            'crs':glacier_crs_str,
                            'resolution': f'{res_mill} m',
                            'timestamp':'2017-2018',
                            'description':'Velocity from Millan et al. (2022). Missing/NaN values were filled with 0.',
                            'nodata': 0
                            })
    
    da_vy = da_vy.rename('VY').assign_attrs({'long_name': 'surface ice velocity (y-component)',
                            'units':'m/year',
                            'crs':glacier_crs_str,
                            'resolution': f'{res_mill} m',
                            'timestamp':'2017-2018',
                            'description':'Velocity from Millan et al. (2022). Missing/NaN values were filled with 0.',
                            'nodata': 0
                            })
    da_stdx = da_stdx.rename('UNCT_VX').assign_attrs({'long_name': 'surface ice velocity uncertainty (x-component)',
                            'units':'m/year',
                            'crs':glacier_crs_str,
                            'resolution': f'{res_mill} m',
                            'timestamp':'2017-2018',
                            'description':'Velocity uncertainty from Millan et al. (2022) (STDX). Missing/NaN values were filled with 0.',
                            'nodata': 0
                            })
    da_stdy = da_stdy.rename('UNCT_VY').assign_attrs({'long_name': 'surface ice velocity uncertainty (y-component)',
                            'units':'m/year',
                            'crs':glacier_crs_str,
                            'resolution': f'{res_mill} m',
                            'timestamp':'2017-2018',
                            'description':'Velocity uncertainty from Millan et al. (2022) (STDY). Missing/NaN values were filled with 0.',
                            'nodata': 0
                            })
    

    ''' --- ## Save dataset to netcdf ---------------- '''
    
    ds_glacier_nc = (xr.Dataset({   'VX': da_vx,
                                    'VY': da_vy,
                                    'UNCT_VX': da_stdx,
                                    'UNCT_VY': da_stdy,
                    }).assign_attrs({'title':'Glacier observation data ofVX and VY from Global Dataset',
                        'grid_resolution': f'{str(da_vx.rio.resolution())} m',
                        'description':'see attributes of each variable',
                        'timestamp':'',
                    })
                    .rio.set_spatial_dims(x_dim="x", y_dim="y") # Make sure spatial dims are known
                    # Write CRS and CF grid mapping to the whole dataset
                    .rio.write_crs(glacier_crs)
                    .rio.write_grid_mapping("spatial_ref")
                    .rio.write_transform()
    )
    
    fname_nc = f'{glacier_Name.lower()}_globaldata_VX-VY-millan.nc'
    save_encoded_nc(ds_glacier_nc, path2data_global, fname_nc, comp=None)


    ''' ----------------------
    ## Millan thickness
    -------------------------- '''
    print('.. loading THK millan')
    tstrt = time.time()
    res_mill = 50 # m : both for velo and thickness
    # da_grid_mill = datafuncs.create_regular_dummy_grid(da_grid_glacier, res=res_mill, resample_method=rio.enums.Resampling.nearest)

    ## get sub-directory for corret RGI region of current glacier
    rgi_dir_options = os.listdir(os.path.join(path2data_input, 'Millan2022/thickness'))
    rgi_dir_millan = [dirname for dirname in rgi_dir_options if f'RGI-{glacier_RGIregions[glacier_Name].lstrip("0")}' in dirname] ## need to be vague because sometimes RGI region number is specific in dirname, sometimes its part of a group
    assert len(rgi_dir_millan) == 1, "Number of Millan sub-directories not 1"
    path2millan = os.path.join(path2data_input, 'Millan2022/thickness', rgi_dir_millan[0])
    thk_list = sorted([f for f in os.listdir(path2millan) if 'THICKNESS' in f])
    thk_err_list = sorted([f for f in os.listdir(path2millan) if 'ERRTHICKNESS' in f])
    ## make full path name
    thk_list = [os.path.join(path2millan, f) for f in thk_list] ## full pathnames
    thk_err_list = [os.path.join(path2millan, f) for f in thk_err_list] ## full pathnames

    ## load files into a single dataArray; 
    if len(thk_list) > 1: ## for some RGI regions there's multiple files, for other regions theres only 1 file.
        da_thk_mill, tilelist_domain_millan_thk = load_combine_tilelist_in_domain(thk_list, da_grid_mill, resample_method=rio.enums.Resampling.nearest)
        da_thk_mill_err,   _ = load_combine_tilelist_in_domain(thk_err_list, da_grid_mill, resample_method=rio.enums.Resampling.nearest)

    elif len(thk_list) == 1:
        da_thk_mill = xr.open_dataarray(thk_list[0]).isel(band=0).drop_vars('band') 
        da_thk_mill_err = xr.open_dataarray(thk_err_list[0]).isel(band=0).drop_vars('band')
        tilelist_domain_millan_thk = thk_list.copy()
        ## reproject and match glacier CRS
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", category=UserWarning) ## ignore 'rectified to skew grid conversion' warnings 
        ## reproject to regularized resolution
        da_thk_mill     = datafuncs.reproject_match_grid(da_grid_mill, da_thk_mill, resample_method=rio.enums.Resampling.nearest)
        da_thk_mill_err = datafuncs.reproject_match_grid(da_grid_mill, da_thk_mill_err, resample_method=rio.enums.Resampling.nearest)
    else:
        raise ValueError(f"Did not find any files for {glacier_Name} in Millan2022 dataset.")

    print('  ', da_thk_mill.rio.crs, da_thk_mill.rio.resolution(), da_thk_mill.shape)
    
    assert da_thk_mill.rio.crs == da_thk_mill_err.rio.crs == glacier_crs, 'CRS of Millan data does not match CRS of glacier data'


    ### assign attributes
    da_thk_mill = da_thk_mill.rename('THK_Mi22').fillna(0).assign_attrs({'long_name':'Ice Thickness',
                                   'units':'m',
                                   'crs':glacier_crs_str,
                                   'resolution': f'{res_mill} m',
                                   'timestamp':'unknown',
                                   'description':'ice thickness from Millan et al. (2022)',
                                   'nodata': 0
                                   })

    

    ''' --- ## Save dataset to netcdf ---------------- '''
    
    ds_glacier_nc = (xr.Dataset({   'THK': da_thk_mill,
                                    'UNCT_THK': da_thk_mill_err,
                    }).assign_attrs({'title':'Glacier observation data of THK from Global Dataset',
                        'grid_resolution': f'{str(da_thk_mill.rio.resolution())} m',
                        'description':'see attributes of each variable',
                        'timestamp':'',
                    })
                    .rio.set_spatial_dims(x_dim="x", y_dim="y") # Make sure spatial dims are known
                    # Write CRS and CF grid mapping to the whole dataset
                    .rio.write_crs(glacier_crs)
                    .rio.write_grid_mapping("spatial_ref")
                    .rio.write_transform()
    )
    
    fname_nc = f'{glacier_Name.lower()}_globaldata_THK-millan.nc'
    save_encoded_nc(ds_glacier_nc, path2data_global, fname_nc, comp=None)


    ###%%
    ''' ----------------------
    ## Farinotti thickness
    - does not have a single thickness resolution. Resolutions for files in our selected glacier domains vary between 
        Aletsch: 25m ; 50, ... --> actual aletsch file is in 50 m . 
        ..
        Hofsjokull: 25m, 50m, 100m for all the 11 different flow basins of the ice cap. --> use 50 m (only one basin has res=100; 4/11 have 25 and 6/11 have 50m)
    --> need to just identify and load sonly the relevant file. Automatic combining goes wonky
    -------------------------- '''
    print('.. loading THK farinotti')

    # # res_fari = ?? # m
    rgi_list_hofsj =  ['RGI60-06.00228','RGI60-06.00229','RGI60-06.00230','RGI60-06.00231','RGI60-06.00232','RGI60-06.00233',
                       'RGI60-06.00234','RGI60-06.00235','RGI60-06.00236','RGI60-06.00237','RGI60-06.00238']
    rgi_list_hofsj =  ['RGI60-06.00228_thickness.tif','RGI60-06.00229_thickness.tif','RGI60-06.00230_thickness.tif','RGI60-06.00231_thickness.tif','RGI60-06.00232_thickness.tif','RGI60-06.00233_thickness.tif',
                       'RGI60-06.00234_thickness.tif','RGI60-06.00235_thickness.tif','RGI60-06.00236_thickness.tif','RGI60-06.00237_thickness.tif','RGI60-06.00238_thickness.tif']
    fari_files = {'Aletsch': {'rgiid':['RGI60-11.01450_thickness.tif'],
                              'res': 50},
                  'Argentiere': {'rgiid':['RGI60-11.03638_thickness.tif'],
                                 'res': 25}, 
                  'Gepatschferner': {'rgiid':['RGI60-11.00746_thickness.tif'],
                                 'res': 25}, 
                  'Hofsjokull': {'rgiid':rgi_list_hofsj,
                                 'res': 50}, 
                  'SaryTor': {'rgiid': ['RGI60-13.08055_thickness.tif'],  # # 'RGI60-13.08056_thickness.tif',  'RGI60-13.08055_thickness.tif' --> what lander said
                                 'res': 25}, 
                  'Zongo': {'rgiid': ['RGI60-16.00543_thickness.tif','RGI60-16.00546_thickness.tif','RGI60-16.00541_thickness.tif' ,'RGI60-16.00540_thickness.tif'],
                                 'res': 25}}

    ## get sub-directory for corret RGI region of current glacier
    rgi_dir_options = os.listdir(os.path.join(path2data_input, 'Farinotti2019'))
    rgi_dir_farinotti = [dirname for dirname in rgi_dir_options if f'RGI60-{glacier_RGIregions[glacier_Name]}' in dirname] 
    assert len(rgi_dir_farinotti) == 1, "Number of Farinotti sub-directories not 1"
    path2farinotti = os.path.join(path2data_input, 'Farinotti2019', rgi_dir_farinotti[0])

    ## set up grid for glacier with specifi resolution
    thk_file_glacier = fari_files[glacier_Name]['rgiid']
    thk_res = fari_files[glacier_Name]['res']
    da_grid_fari_glacier = datafuncs.create_regular_dummy_grid(da_grid_glacier, grid_res=thk_res, crs=glacier_crs, unit='m', add_buffer=None)

    
    if len(thk_file_glacier) > 1: ## for hofsjokull and Zongo
        print('.. more than 1 file')
        tilelist_domain_fari = [os.path.join(path2farinotti, f) for f in thk_file_glacier]

        da_datatiles_list = []
        for tile_file in tilelist_domain_fari:
            # # tile_file = os.path.join(path2farinotti, f"{rgiid_file}_thickness.tif")
            # tile_file = os.path.join(path2farinotti, rgiid_file)
            with xr.open_dataarray(tile_file).isel(band=0).drop_vars('band') as da_tile:
                ''' make sure file has valid CRS'''
                if not da_tile.rio.crs:
                    raise ValueError(f"CRS is missing for {tile_file}. Cannot reproject without a valid CRS.")
                ''' ## rerpoject to domain grid '''
                ## hofsjokul: remove 0 values so combining files works well
                da_tile = da_tile.where(da_tile>0)
                # print('tile res : ', da_tile.rio.resolution())
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning) ## ignore 'rectified to skew grid conversion' warnings
                    da_tile_reprj = datafuncs.reproject_match_grid(da_grid_fari_glacier, da_tile, resample_method=rio.enums.Resampling.nearest)
                    da_datatiles_list.append(da_tile_reprj)
        
        ## For these we can use combine_first (after 0 values have been removed)
        da_thk_fari = da_datatiles_list[0] # initialise domain wide dataset contianing 1 region of data
        for ds in da_datatiles_list[1:]:
            # combine_first() defaults to non-null values in the calling object, and fills holes with called object.
            # effecitvely patching all regions to the first
            da_thk_fari = da_thk_fari.combine_first(ds) 
    
    elif len(thk_file_glacier) == 1:
        tilelist_domain_fari = [os.path.join(path2farinotti, f) for f in thk_file_glacier]
        
        ''' ## load single glacier RGIID file'''
        thk_file_glacier = os.path.join(path2farinotti, thk_file_glacier[0])
        da_thk_fari = xr.open_dataarray(thk_file_glacier).isel(band=0).drop_vars('band')
        ''' ## rerpoject to domain grid '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning) ## ignore 'rectified to skew grid conversion' warnings
            da_thk_fari = datafuncs.reproject_match_grid(da_grid_fari_glacier, da_thk_fari, resample_method=rio.enums.Resampling.nearest)   
    
    else:
        raise ValueError(f"Did not find any files for {glacier_Name} in Farinotti2019 dataset.")
    
    assert da_thk_fari.rio.crs == glacier_crs, 'CRS of Farinotti data does not match CRS of glacier data'


    ### assign attributes
    da_thk_fari = da_thk_fari.rename('THK_Fa19').fillna(0).assign_attrs({'long_name':'Ice Thickness',
                                    'units':'m',
                                    'resolution': f'{thk_res} m',
                                    'crs':glacier_crs_str,
                                    'timestamp':'unknown',
                                    'description':'ice thickness from Farinotti et al. (2019)',
                                    'nodata': 0
                                    })
    

    ''' --- ## Save dataset to netcdf ---------------- '''
    
    ds_glacier_nc = (xr.Dataset({
                                    'THK': da_thk_fari,
                    }).assign_attrs({'title':'Glacier observation data of THK from Global Dataset',
                        'grid_resolution': f'{str(da_thk_fari.rio.resolution())} m',
                        'description':'see attributes of each variable',
                        'timestamp':'',
                    })
                    .rio.set_spatial_dims(x_dim="x", y_dim="y") # Make sure spatial dims are known
                    # Write CRS and CF grid mapping to the whole dataset
                    .rio.write_crs(glacier_crs)
                    .rio.write_grid_mapping("spatial_ref")
                    .rio.write_transform()
    )
    
    fname_nc = f'{glacier_Name.lower()}_globaldata_THK-farinotti.nc'
    save_encoded_nc(ds_glacier_nc, path2data_global, fname_nc, comp=None)

    
    ''' ----------------------
    ## Maffezzoli thickness
    - this dataset is saved per RGIId of all glaciers. Meaning some may overlap.
    -------------------------- '''
    tstrt = time.time()
    print('.. loading THK maffezzoli')

    ## get sub-directory for corret RGI region of current glacier
    rgi_dir_options = os.listdir(os.path.join(path2data_input, 'Maffezzoli2025'))
    rgi_dir_maff = [dirname for dirname in rgi_dir_options if f'rgi{glacier_RGIregions[glacier_Name].lstrip("0")}' in dirname] ## need to be vague because sometimes RGI region number is specific in dirname, sometimes its part of a group
    assert len(rgi_dir_maff) == 1
    path2maff = os.path.join(path2data_input, 'Maffezzoli2025', rgi_dir_maff[0])
    thk_list = sorted(os.listdir(path2maff))
    print(len(thk_list))
    ## refine with known files
    thk_list = [f for f in thk_list if f in filelist_maff]
    print(len(thk_list))

    thk_list = [os.path.join(path2maff, f) for f in thk_list] ## full pathnames

    ''' Maffezzolli thickness tiles should not be combined using xr.combine_first, because some tiles overlap with each other.
    Instead, concatenate along a third dimension and then take the mean or median of the stacked tiles. 
    This assumes that the overlapping pixels will have the same thickness value
    '''
    
    tilelist_domain_maff = thk_list.copy() 
    # print(f"Found {len(tilelist_domain_maff)} tiles in {glacier_Name} domain")

    da_datatiles_list = []
    for tile_file in tilelist_domain_maff:
        with xr.open_dataarray(tile_file).isel(band=0).drop_vars('band') as da_tile:
            ''' make sure file has valid CRS'''
            if not da_tile.rio.crs:
                raise ValueError(f"CRS is missing for {tile_file}. Cannot reproject without a valid CRS.")

            ''' ## rerpoject to domain grid '''
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning) ## ignore 'rectified to skew grid conversion' warnings
           
                da_tile_reprj = datafuncs.reproject_match_grid(da_grid_glacier, da_tile, resample_method=rio.enums.Resampling.nearest)
                da_datatiles_list.append(da_tile_reprj)
                    
    ##%% stack along third dim
    da_tilestack = xr.concat(da_datatiles_list, dim='tile')
    da_thk_maff = da_tilestack.mean(dim='tile', skipna=True) # take mean of stacked tiles, ignoring nans

    ### assign attributes
    da_thk_maff = da_thk_maff.rename('THK_M25').fillna(0).assign_attrs({'long_name':'Ice Thickness',
                                   'units':'m',
                                   'crs':glacier_crs_str,
                                   'timestamp':'unknown',
                                   'description':'ice thickness from Maffezzoli et al. (2025)',
                                   'nodata': 0
                                   })
    
    # fig,ax=plt.subplots(figsize=(6,5))
    # da_thk_maff.plot.imshow(ax=ax, vmin=0, vmax=300, cmap='Blues', cbar_kwargs={'shrink': 0.7})
    # ax.set_title(f'thk {glacier_Name}')
    # ax.set_aspect('equal')

    ''' --- ## Save dataset to netcdf ---------------- '''
    
    ds_glacier_nc = (xr.Dataset({
                                    'THK': da_thk_maff,
                    }).assign_attrs({'title':'Glacier observation data of THK from Global Dataset',
                        'grid_resolution': f'{str(da_thk_maff.rio.resolution())} m',
                        'description':'see attributes of each variable',
                        'timestamp':'',
                    })
                    .rio.set_spatial_dims(x_dim="x", y_dim="y") # Make sure spatial dims are known
                    # Write CRS and CF grid mapping to the whole dataset
                    .rio.write_crs(glacier_crs)
                    .rio.write_grid_mapping("spatial_ref")
                    .rio.write_transform()
    )
    
    fname_nc = f'{glacier_Name.lower()}_globaldata_THK-maffezzoli.nc'
    save_encoded_nc(ds_glacier_nc, path2data_global, fname_nc, comp=None)

    


#%%

'''--------------------
Plot variables for each glacier
-----------------------'''

files_global = sorted(os.listdir(path2data_global))

for glaciername in glacier_names:
    files_glacier = [f for f in files_global if f.startswith(glaciername.lower()) and f.endswith('.nc')]
    fname_THKs = sorted([f for f in files_glacier if 'THK' in f]) ## farinotti, maffezoli, millan
    fname_vels = [f for f in files_glacier if 'VX' in f ]
    fname_dhdt = [f for f in files_glacier if 'DHDT' in f]
    
    da_thk_fari = xr.open_dataset(os.path.join(path2data_global, fname_THKs[0]),decode_coords='all')['THK']
    da_thk_maff = xr.open_dataset(os.path.join(path2data_global, fname_THKs[1]),decode_coords='all')['THK']
    da_thk_mill = xr.open_dataset(os.path.join(path2data_global, fname_THKs[2]),decode_coords='all')['THK']
    da_vx = xr.open_dataset(os.path.join(path2data_global, fname_vels[0]),decode_coords='all')['VX']
    da_vy = xr.open_dataset(os.path.join(path2data_global, fname_vels[0]),decode_coords='all')['VY']
    da_dhdt = xr.open_dataset(os.path.join(path2data_global, fname_dhdt[0]),decode_coords='all')['DHDT']
            

    ## check values by plotting
    fig,axs=plt.subplots(2,3, figsize=(10,8))
    row,col = 0,0
    for da_plot, var, cmap, vminmax in zip(  [da_thk_fari, da_thk_maff, da_thk_mill, da_dhdt, da_vx, da_vy],
                                    ['THK-farinotti', 'THK-maffezzoli', 'THK-millan',   'DHDT-hugonnet',     'VX-millan', 'VY-millan', ],
                                    [ 'Blues',      'Blues',     'Blues',             'RdBu',     'PiYG','PiYG'],
                                    [(0,400),(0,400),(0,400),                   (-5,5), (-100,100),(-100,100)]):
        if vminmax is not None:
            vmin, vmax = vminmax
        else:
            vmin, vmax = None, None

        ax=axs[row,col]
        da_plot.plot.imshow(ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, cbar_kwargs={'shrink': 0.7})
        ax.set_title(var)
        col+=1
        if col >= 3:
            col = 0
            row += 1
    [ax.set_aspect('equal') for ax in axs.flatten()];
    [ax.set_axis_off() for ax in axs.flatten()];

    fig.savefig(os.path.join(path2data_global, f'{glaciername}_global_vars.png'), dpi=300)

# %%
