#%% Check and pre-processing of data
''' Preprocessing of observational data for the Gepatschferner glacier.
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
import glob

path2data_raw = '../../ContinuIX_WP1_data/Data_Package/01_submitted_data/Gepatschferner/'
path2data_clean = '../../ContinuIX_WP1_data/Data_Package/02_raw-cleaned_data/Gepatschferner/'
path2data_homog = '../../ContinuIX_WP1_data/Data_Package/03_homogenized_data/Gepatschferner/'

import datafunctions as datafuncs


#%% Step 0: Check submitted (raw) data
''' ##################################
Check submitted (raw) data where needed

## dhdt: in meter --> convert to m/yr

## outlines: geojson to shapefile

## DEM: downloaded from data.tirol.gv.at; is in EPSG:31254 so should reproject to 25832
##################################
'''


gdf_2006 = gpd.read_file(os.path.join(path2data_raw, 'gepatsch_outline_2006.geojson'))
gdf_2017 = gpd.read_file(os.path.join(path2data_raw, 'gepatsch_outline_2017.geojson'))

assert gdf_2006.crs == gdf_2017.crs == 'EPSG:25832', "CRS of outlines do not match"

## save to shapefile in cleaned dir

fname = f'gepatsch_outline_2006.shp'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    print(f'Saving {fname} to cleaned data directory...')
    gdf_2006.to_file(os.path.join(path2data_clean, fname), driver='ESRI Shapefile')
else:
    print(f"File {fname} already exists in cleaned data directory. Skipping save.")

fname = f'gepatsch_outline_2017.shp'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    print(f'Saving {fname} to cleaned data directory...')
    gdf_2017.to_file(os.path.join(path2data_clean, fname), driver='ESRI Shapefile')
else:
    print(f"File {fname} already exists in cleaned data directory. Skipping save.")



#%%
''' ########################
### dhdt
############################ '''
da_bedrock = xr.open_dataarray(os.path.join(path2data_raw, 'gepatsch_bedrock_2006.tif')
                               ).isel(band=0).drop_vars('band') ## 10 m


dh_1 = xr.open_dataarray(os.path.join(path2data_raw, 'gepatsch_dh_2006-2017.tif')
                             ).isel(band=0).drop_vars('band')

dh_2 = xr.open_dataarray(os.path.join(path2data_raw, 'gepatsch_dh_2006-2018.tif')
                             ).isel(band=0).drop_vars('band')

dh_1.rio.resolution() ## 1 m

## reproject to bedrock grid resolution (10 m)
da_dh_1 = datafuncs.reproject_match_grid(da_bedrock, dh_1, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
da_dh_2 = datafuncs.reproject_match_grid(da_bedrock, dh_2, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
da_dh = da_dh_1.combine_first(da_dh_2)

dt = 2017-2006
da_dhdt = da_dh / dt ## in m/yr

## clip to outline 
da_dhdt = da_dhdt.rio.clip(gdf_2006.geometry, gdf_2006.crs, drop=False) # drop=False to keep the same grid and not drop the pixels outside the outline (which will be set to nodata)

## store to cleaned data folder
fname = f'gepatsch_dhdt_2006-2017.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    print(f'Saving {fname} to cleaned data directory...')
    da_dhdt.rio.to_raster(os.path.join(path2data_clean, fname))
else:
    print(f"File {fname} already exists in cleaned data directory. Skipping save.")


# %%
''' ########################
### DEMs: from data.tirol.gv.at; is in 50cm resolution
1. combine DEMs at lower resolution (2m), in EPSG25832, save in 'submitted' folder
2. no further cleaning needed
3. reproject to 10m resolution for 'homogenized'
############################ '''
import glob 

file_to_dem = '../../ContinuIX_WP1_data/11_Gepatschferner/tirol-gov_DEM-DTM/DOM_DEM/'
filelist_dem17 = glob.glob(os.path.join(file_to_dem, '*.tif'))
# ds_DEM_2017 = xr.mf

## create dummy grid
da_dummy_2m = datafuncs.create_regular_dummy_grid(da_dhdt, grid_res=2, crs=25832, unit='m')
da_dem_list = []
for file in filelist_dem17:
    da_tile = xr.open_mfdataset(file).isel(band=0).drop_vars('band')['band_data']

    da_tile_2m = datafuncs.reproject_match_grid(da_dummy_2m, da_tile, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
    da_dem_list.append(da_tile_2m)

## combine DEMs at 2m resolution
da_dem_2017_2m = xr.combine_nested(da_dem_list, concat_dim='band').max(dim='band') ## patches all tiles and taking max means taking the single non-nan value.
# da_dem_2017_2m.plot.imshow()

assert da_dem_2017_2m.rio.crs == da_dhdt.rio.crs, "CRS of DEM and dummy grid do not match"
## save to submitted data folder
fname = 'gepatsch_DEM_2017.tif'
if not os.path.exists(os.path.join(path2data_raw, fname)):
    print(f'Saving {fname} to submitted data directory...')
    da_dem_2017_2m.rio.to_raster(os.path.join(path2data_raw, fname))
else: print(f"File {fname} already exists in submitted data directory. Skipping save.")

'''## --- do the same for tirol.gov BED (DTM)
--> no not needed, DTM and DEM are pracically the same '''

file_to_dtm = '../../ContinuIX_WP1_data/11_Gepatschferner/tirol-gov_DEM-DTM/DGM_DTM/'
filelist_dtm17 = glob.glob(os.path.join(file_to_dtm, '*.tif'))
da_dtm_list = []
for file in filelist_dtm17:
    da_tile = xr.open_mfdataset(file).isel(band=0).drop_vars('band')['band_data']

    da_tile_2m = datafuncs.reproject_match_grid(da_dummy_2m, da_tile, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
    da_dtm_list.append(da_tile_2m)

da_dtm_2017_2m = xr.combine_nested(da_dtm_list, concat_dim='band').max(dim='band') ## patches all tiles and taking max means taking the single non-nan value.
da_dtm_2017_2m.plot.imshow()

# fname = 'gepatsch_DTM_2017.tif'
# if not os.path.exists(os.path.join(path2data_raw, fname)):
#     print(f'Saving {fname} to submitted data directory...')
#     da_dtm_2017_2m.rio.to_raster(os.path.join(path2data_raw, fname),driver='COG')
# else: print(f"File {fname} already exists in submitted data directory. Skipping save.")

''' ## -----
Also save both to CLEANED dir. Rename 'DTM to 'bedrock'
----- '''
fname = 'gepatsch_DEM_2017.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    print(f'Saving {fname} to cleaned data directory...')
    da_dem_2017_2m.rio.to_raster(os.path.join(path2data_clean, fname))
else: print(f"File {fname} already exists in cleaned data directory. Skipping save.")

# fname = 'gepatsch_bedrock_2017.tif'
# if not os.path.exists(os.path.join(path2data_clean, fname)):
#     print(f'Saving {fname} to cleaned data directory...')
#     da_dtm_2017_2m.rio.to_raster(os.path.join(path2data_clean, fname),driver='COG')
# else: print(f"File {fname} already exists in cleaned data directory. Skipping save.")


# #%%
# ''' ### check bedrock of DTM to the provided bedrock_2006
# - ok so what I thought that DTM=bedrock is not true --> rely on bedrock_2006 which is what they actually submitted.
# - apparently DTM == DEM  practically.. 
#     '''
# fig,axs=plt.subplots(1,3, figsize=(12,6))
# ax=axs[0]
# da_dem_2017_2m.plot.imshow(ax=ax, cmap='terrain', vmin=2000, vmax=4000)
# ax=axs[1]
# da_dtm_2017_2m.plot.imshow(ax=ax, cmap='terrain', vmin=2000, vmax=4000)
# da_dtm_10m = datafuncs.reproject_match_grid(da_bedrock, da_dtm_2017_2m, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
# bedrock_diff = da_dtm_10m - (da_bedrock + datafuncs.reproject_match_grid(da_bedrock, da_thickness, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan) )
bedrock_diff = (da_dtm_2017_2m - da_dem_2017_2m).rio.clip(gdf_2006.geometry)
median_diff = bedrock_diff.median().item()
stddev_diff = bedrock_diff.std().item()

# ax=axs[2]
# bedrock_diff.plot.imshow(ax=ax, cmap='RdBu', vmin=-5, vmax=5)




# %% Looking at stake veloicity
''' ########################
### STAKE VELOCITY

# from readme:
- each row represents coordinates between consequetive readings (date0 to date1; or date0 to datemid to date1).
- In rows where coordinates were filled only for date0 and date1, the distance between those coordinates can be used to derive an annual velocity. 
- If coordinates for "datemid" were filled, the annual velocity should be computed by combining the first period with the second 
    (for example, distance moved in X direction: (X0-X0_m)+(X1_m-X1)). 
    NB UPDATE: this was not specified correctly in the readme. Should be (X0_m-X0)+(X1-X1_m) to get the correct direction of movement. Same for Y direction.


--> so e.g. stake P60:
    P60 row 1  : t0 --> t1
    P60 row 2  : t2 --> t3
    --> also get cross-row displacement/velocity? so t1-->t2? or not, since that's not instructed from the readme.
--> also: x0-x0_m seems weird, why not do x0_m-x0, since we want to get the velocity in the same direction as the movement (from t0 to t1)? 

--> for velo from distance: need to do same mid-step as distance, or just overall? should be same right
############################ '''

df_coords = pd.read_csv(os.path.join(path2data_raw, 'gepatsch_coordinates_2009_2024.csv'))
df_coords
df_crs = 'EPSG:25832' ## from readme

## convert date0, datemid, date1 to datetime
## rename columns
df_coords.rename(columns={'date0':'date0_str', 'datemid':'datemid_str', 'date1':'date1_str'}, inplace=True)
df_coords['date0'] = pd.to_datetime(df_coords['date0_str'], format='%d.%m.%Y', errors='coerce')
# df_coords['datemid'] = pd.to_datetime(df_coords['datemid_str'], format='%d.%m.%Y', errors='coerce')
df_coords['datemid'] = pd.to_datetime(df_coords['datemid_str'], format='%d.%m.%Y %H:%M')
df_coords['date1'] = pd.to_datetime(df_coords['date1_str'], format='%d.%m.%Y', errors='coerce')
df_coords['duration_days'] = (df_coords['date1'] - df_coords['date0']).dt.days
df_coords['dt_yr'] = df_coords['duration_days'] / 365.25 ## in years
df_coords

## PLOT SETTINGS
scale_factor = 5
norm_v = plt.Normalize(vmin=0, vmax=60)
cmap_v = plt.cm.plasma
cmap_v = plt.cm.cividis

colors_stakes = plt.cm.tab20#[0:df_coords['stake'].nunique()] # tab20 has 20 colors, so if more than 20 stakes, will need to use a different colormap or cycle through colors
norm_stakes = plt.Normalize(vmin=0, vmax=20)
norm_yr = plt.Normalize(vmin=df_coords['date0'].min().year, vmax=df_coords['date0'].max().year)


fig, ax = plt.subplots(figsize=(8, 8))
gdf_2006.plot(ax=ax, facecolor='steelblue', edgecolor='steelblue', # lightblue
                    linewidth=1.5, alpha=0.5, label='Glacier outline 2006')
gdf_2006.boundary.plot(ax=ax, edgecolor='steelblue', linewidth=1.5, alpha=0.5, label='Glacier outline 2006')

stake_i = 0
# # for df_stake in df_coords.groupby('stake'):
# #     print(df_stake)
    
#     stake_id = df_stake[0]
#     df_stake = df_stake[1]


# ## DEV: check one unique stake. Choose P66.
stake_ids = ['P60 (P51)', 'P61', 
             'P62 (P52)', 'P63', 'P64', 
             'P65 (P53)', 'P66', 'P67', 
             'P68', 'P69', 'P70', 'P71',
             'P72', 'P73', 'P74', 'P75']

for stake_id in stake_ids:
    # df_stake = df_coords[df_coords['stake'] == 'P66']
    df_stake = df_coords[df_coords['stake'] == stake_id]
    df_stake

    print(f"Stake {stake_id}:")
    # print(df_stake)
    # print("\n")


    '''## (1) calculate as specified in readme '''
    # ## (A) calculate distance between date0 and date1
    # df_stake['dx_01'] = df_stake['X0'] - df_stake['X1']
    # df_stake['dy_01'] = df_stake['Y0'] - df_stake['Y1']

    # ## (B) calculate distance between date0 and datemid, and datemid and date1
    # df_stake['dx_0m1'] = (df_stake['X0'] - df_stake['X0_m']) + (df_stake['X1'] - df_stake['X1_m'])
    # df_stake['dy_0m1'] = (df_stake['Y0'] - df_stake['Y0_m']) + (df_stake['Y1'] - df_stake['Y1_m'])

    '''## (2) calculate as I think it should'''
    ## (A) calculate distance between date0 and date1
    df_stake['dx_01'] = df_stake['X1'] - df_stake['X0']
    df_stake['dy_01'] = df_stake['Y1'] - df_stake['Y0']

    ## (B) calculate distance between date0 and datemid, and datemid and date1
    df_stake['dx_0m1'] = (df_stake['X0_m'] - df_stake['X0']) + (df_stake['X1'] - df_stake['X1_m'])
    df_stake['dy_0m1'] = (df_stake['Y0_m'] - df_stake['Y0']) + (df_stake['Y1'] - df_stake['Y1_m'])



    ''' ## ---'''

    ## select which distance to use: if datemid is not null, use dx_0m1 and dy_0m1, otherwise use dx_01 and dy_01
    df_stake['dx'] = np.where(df_stake['datemid'].notnull(), df_stake['dx_0m1'], df_stake['dx_01'])
    df_stake['dy'] = np.where(df_stake['datemid'].notnull(), df_stake['dy_0m1'], df_stake['dy_01'])
    df_stake

    ## convert to velocity 
    # df_stake['distance_m'] = np.sqrt(df_stake['dx']**2 + df_stake['dy']**2)
    # df_stake['velocity_myr'] = df_stake['distance_m'] / df_stake['dt_yr']

    df_stake['vx_myr'] = df_stake['dx'] / df_stake['dt_yr']
    df_stake['vy_myr'] = df_stake['dy'] / df_stake['dt_yr']
    df_stake['velocity_myr'] = np.sqrt(df_stake['vx_myr']**2 + df_stake['vy_myr']**2)

    ## angle of movement in degrees 
    angle_rad = np.arctan2(df_stake['dy'], df_stake['dx'])
    angle_deg = np.degrees(angle_rad) ## from -180 to 180 ; # 0: east, 90: north, 180 or -180: west, -90: south
    angle_deg_360 = (angle_deg + 360) % 360 ## from 0 to 360
    # 0°   = east / +x
    # 90°  = north / +y
    # 180° = west / -x
    # 270° = south / -y
    df_stake['angle'] = angle_deg_360

    ## SUBSELECT columns without datemid
    # df_plot = df_stake.dropna(subset=['datemid'])
    # df_plot = df_stake.loc[~df_stake['datemid'].isna(),:]    ## rows where datemid ISNOT nan

    ### %% PLOT stakes

    # fig, ax = plt.subplots(figsize=(8, 8))
    # gdf_2006.plot(ax=ax, facecolor='steelblue', edgecolor='steelblue', # lightblue
    #                     linewidth=1.5, alpha=0.5, label='Glacier outline 2006')

    ## arrow per stake
    # x = angle['x']
    # y = angle['y']   

    # arrow_length = 150
    # # arrow_length = dmg*8000 # scale lenngth with dmg
    # arrow_length = 250 + dmg*3000
    arrow_length = 100 # 300+dmg*3000
    scale = 500
    width=8
    headwidth=4


    '''## define arrow
    # 'xy': Arrow direction in data coordinates, 
            i.e. the arrows point from (x, y) to (x+u, y+v). 
            This is ideal for vector fields or gradient plots where the arrows 
            should directly represent movements or gradients in the x and y directions.
    # 'uv'; Arrow directions are based on display coordinates; 
            i.e. a 45° angle will always show up as diagonal on the screen, irrespective of figure or Axes aspect ratio or Axes data ranges. 
            This is useful when the arrows represent a quantity whose direction is not tied to the x and y data coordinates.
    '''
    # x = df_stake['X0'].values
    # y = df_stake['Y0'].values
    # # u = df_stake['vx_myr'].values ## using DX or VX_MYR gives the same plot
    # # v = df_stake['vy_myr'].values
    # u = df_stake['dx'].values ## using DX or VX_MYR gives the same plot
    # v = df_stake['dy'].values
    # ax.quiver(x,y, u,v, 
    #             angles='xy',
    #             units='x',
    #             # scale=0., #1.1,#0.8,#0.4,
    #             scale_units='width', 
    #             # scale_units='x', # sclae_unit 'x': unit will be 0.5 xax units -- setting scale=1 means 0.5xaxunit, scale > 1:  means <0.5xaxunit (smaller), scale=0-1: bigger
    #             width=width, #120,#100, # 0.005 typical starting value
    #             # headwidth=headwidth,#4, # default 3
    #             pivot='tail',
    #             color=colors_stakes(stake_i), # color by stake id
    #             )
    # ax.set_title('Stake velocity using x,y and angles=''xy'' ')
    
    # # angletype = 'uv' -- DONT USE !!
    # # x = df_stake['X0'].values
    # # y = df_stake['Y0'].values
    # # u = arrow_length * np.cos(np.deg2rad(df_stake['angle'].values) )
    # # v = arrow_length * np.sin(np.deg2rad(df_stake['angle'].values) )
    # # ax.quiver(x,y, u,v, 
    # #             angles='uv',
    # #             units='x',
    # #             # scale=scale, #1.1,#0.8,#0.4,
    # #             scale_units='x', # sclae_unit 'x': unit will be 0.5 xax units -- setting scale=1 means 0.5xaxunit, scale > 1:  means <0.5xaxunit (smaller), scale=0-1: bigger
    # #             width=width, #120,#100, # 0.005 typical starting value
    # #             # headwidth=headwidth,#4, # default 3
    # #             # pivot='tail',
    # #             color=colors_stakes(stake_i), # color by stake id
    # #             )
    # # ax.set_aspect('equal')
    # # fig.tight_layout()
    # # ax.set_title('Stake velocity using x,y and angles=''uv'' ')

    # # Arrows per stake
    for _, row in df_stake.iterrows():
        # if pd.isna(row['X0']) or pd.isna(row['Y0']):
        #     continue
        # dist = np.hypot(row['dx'], row['dy'])
        # if dist == 0:
        #     continue
        color = cmap_v(norm_v(row['velocity_myr']))
        norm = norm_v

        # color = cmap(norm_yr(row['date0'].year))
        # norm = norm_yr

        # ux = row['vx_myr'] * scale_factor
        # uy = row['vy_myr'] * scale_factor
        arrow_scale = 5
        # ux = row['dx'] / row['dx'] * row['velocity_myr'] * arrow_scale
        # uy = row['dy'] / row['dy'] * row['velocity_myr'] * arrow_scale
        ux = row['vx_myr'] * arrow_scale
        uy = row['vy_myr'] * arrow_scale

        # x = row['X0']
        # y = row['Y0']
        # arrow_length = 300 + row['velocity_myr'] * scale
        # u = arrow_length * np.cos(np.deg2rad(row['angle']))
        # v = arrow_length * np.sin(np.deg2rad(row['angle']))

        arrow_length = 1 
        # u = arrow_length * row['vx_myr'] 
        # v = arrow_length * row['vy_myr']

        x = row['X0']
        y = row['Y0']
        # u = row['vx_myr'] ## using DX or VX_MYR gives the same plot
        # v = row['vy_myr']
        u = row['dx'] * arrow_length ## scale the dx component to make the arrow length proportional to the velocity, while keeping the angle the same
        v = row['dy'] * arrow_length ## scale the dy component to make the arrow length proportional to the velocity, while keeping the angle the same
        # u = 
        
        # ax.scatter(row['X0'], row['Y0'],
        #             color=color, s=20, zorder=5,
        #             edgecolors='black', linewidths=0.5)
        ax.annotate('',
            xy=(row['X0'] + ux, row['Y0'] + uy),
            xytext=(row['X0'], row['Y0']),
            arrowprops=dict(arrowstyle='->', color=color, lw=1.0),
            zorder=5)

        # ax.quiver(x,y, u,v,
        #             angles='xy',
        #             units='x',
        #             # scale=0.00001, #1.1,#0.8,#0.4,
        #             scale_units='x', # sclae_unit 'x': unit will be 0.5 xax units -- setting scale=1 means 0.5xaxunit, scale > 1:  means <0.5xaxunit (smaller), scale=0-1: bigger
        #             width=width, #120,#100, # 0.005 typical starting value
        #             # headwidth=headwidth,#4, # default 3
        #             pivot='tail',
        #             color=color,
    #                 )
    # # Colorbar
    # sm = plt.cm.ScalarMappable(cmap=cmap_v, norm=norm_v)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    # cbar.set_label('Velocity (m yr⁻¹)')

    # '''# Zoom to stake area'''
    # padding = 500
    # ax.set_xlim(df_stake['X0'].min() - padding,
    #             df_stake['X0'].max() + padding)
    # ax.set_ylim(df_stake['Y0'].min() - padding,
    #             df_stake['Y0'].max() + padding)


    ax.set_xlabel('X / Easting (m)')
    ax.set_ylabel('Y / Northing (m)')
    ax.grid(linestyle='--', alpha=0.3)
    # ax.set_title(f'Stake {stake_id}')

    stake_i += 1

# # Colorbar velo
sm = plt.cm.ScalarMappable(cmap=cmap_v, norm=norm_v)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Velocity (m yr⁻¹)')

# ## colorbar stakes
# # # Colorbar
# sm = plt.cm.ScalarMappable(cmap=colors_stakes, norm=norm_stakes)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
# cbar.set_label('Stake IDs'); cbar.set_ticks(np.arange(0.5, len(stake_ids)+0.5)); cbar.set_ticklabels(stake_ids)

# '''# Zoom to stake area: overall '''
padding = 500
ax.set_xlim(df_coords['X0'].min() - padding,
            df_coords['X0'].max() + padding)
ax.set_ylim(df_coords['Y0'].min() - padding,
            df_coords['Y0'].max() + padding)



ax.set_xlabel('X / Easting (m)')
ax.set_ylabel('Y / Northing (m)')
ax.grid(linestyle='--', alpha=0.3)
ax.set_title(f'Stake velocities')

#%%
''' ########################
### VELOCITY FIELD

# Use Rabatel-2023 velocity; both Rabatel and Millan velocity have OK agreement with stake velocity, but Rabatel offers better uncertainty assessment in accumulation area
- Rabatel flag 'bad' velocity values. In the accumulation area of Gepatsch there's a blob(s) with unrealisticly high velocities 
--> provide velo field and flags for CLEAN data
--> remove and re-interpolate bad values for homogenized data

## SUBMITTED:
- load all rabatell velocity maps of ALPS --> clip to glacier
- include stdev and flags
- in native CRS, 32632 (50 m res)
- annual velocity between 2016-2021

## RAW/CLEAN:
- decompose VELO and DIR to VX and VY (DIR = Flow direction represented as the angle between the flow vector and x-axis. (0°: North))
- reprojected velocity (and flags, stdev, etc.) to 25832
- keep flag / high velo blobs
- keep annual velocity files (if they look good): SELECT only velocities 2015-16 and 2016-17 as they are in dhdt range

## HOMOGENIZED:
- remove flagged high velo blobs, re-interpolate and fill gaps
- merge annual to single velocity field

############################ '''
# target_crs = 'EPSG:25832'
## rabatel velocity is in EPSG:32632

## Gepatsch extent
gepatsch_bounds = da_dhdt.rio.bounds()
## add a buffer of 1 km
gepatsch_bounds = (gepatsch_bounds[0]-1000, gepatsch_bounds[1]-1000, gepatsch_bounds[2]+1000, gepatsch_bounds[3]+1000)
gepatsch_bounds_32632 = rio.warp.transform_bounds(da_dhdt.rio.crs, 'EPSG:32632', *gepatsch_bounds)

'''## Load velocity files, clip to Gepatsch'''
velolist_rabatel = glob.glob('../../ContinuIX_WP1_data/other_data/rabatel_etal_2023_alps_velocity/v*_ALPES_ALL_ANNUALv*.tiff')
# da_v_rabatel = xr.open_mfdataset('../../ContinuIX_WP1_data/other_data/rabatel_etal_2023_alps_velocity/v2015_2016_ALPES_ALL_ANNUALv2016-2021.tiff').isel(band=0)['band_data'].drop_vars('band') ## rabatel velocity is in EPSG:32632

## rabatell velocity is in EPSG:32632
for years_velo in ['2015_2016', '2016_2017', '2017_2018', '2018_2019', '2019_2020', '2020_2021']:
     ## save to RAW folder
    fname = f'gepatsch_v_{years_velo}_rabatel_EPSG32632.tif'
    if not os.path.exists(os.path.join(path2data_raw, fname)):
        with xr.open_mfdataset(f'../../ContinuIX_WP1_data/other_data/rabatel_etal_2023_alps_velocity/v{years_velo}_ALPES_ALL_ANNUALv2016-2021.tiff'
                            ) as da_v_rabatel_year:
            print(f'.. years_velo: {years_velo}')
            
            da_v_rabatel_year = da_v_rabatel_year.isel(band=0)['band_data'].drop_vars('band') ## rabatel velocity is in EPSG:32632
            assert da_v_rabatel_year.rio.crs == 'EPSG:32632', "Rabatel velocity is not in EPSG:32632"

            ## clip to gepatsch bounds
            da_v_rabatel_clip = da_v_rabatel_year.rio.clip_box(*gepatsch_bounds_32632, crs='EPSG:32632')

        
            print(f'Saving {fname} to raw data directory...')
            da_v_rabatel_clip.rio.to_raster(os.path.join(path2data_raw, fname))
    else: print(f"File {fname} already exists in raw data directory. Skipping save.")

'''## same for velocity direction, velocity stdev , flgas'''
da_dir_rabatel = xr.open_mfdataset('../../ContinuIX_WP1_data/other_data/rabatel_etal_2023_alps_velocity/dir_ALPES_ALL_ANNUALv2016-2021.tiff').isel(band=0)['band_data'].drop_vars('band')
da_stdev_rabatel = xr.open_mfdataset('../../ContinuIX_WP1_data/other_data/rabatel_etal_2023_alps_velocity/stdev_ALPES_ALL_ANNUALv2016-2021.tiff').isel(band=0)['band_data'].drop_vars('band')
da_flags_rabatel = xr.open_mfdataset('../../ContinuIX_WP1_data/other_data/rabatel_etal_2023_alps_velocity/flag_ALPES_ALL_ANNUALv2016-2021.tiff').isel(band=0)['band_data'].drop_vars('band')
## assert CRS
assert da_dir_rabatel.rio.crs == da_stdev_rabatel.rio.crs == da_flags_rabatel.rio.crs == 'EPSG:32632', "Rabatel velocity direction is not in EPSG:32632"
## clip bounds
da_dir_rabatel = da_dir_rabatel.rio.clip_box(*gepatsch_bounds_32632, crs='EPSG:32632')
da_stdev_rabatel = da_stdev_rabatel.rio.clip_box(*gepatsch_bounds_32632, crs='EPSG:32632')
da_flags_rabatel = da_flags_rabatel.rio.clip_box(*gepatsch_bounds_32632, crs='EPSG:32632')

## save to SUBMITTED folder
fname = f'gepatsch_velo-dir_rabatel_EPSG32632.tif'
if not os.path.exists(os.path.join(path2data_raw, fname)):
    print(f'Saving {fname} to raw data directory...')
    da_dir_rabatel.rio.to_raster(os.path.join(path2data_raw, fname))
else: print(f"File {fname} already exists in raw data directory. Skipping save.")

fname = f'gepatsch_velo-stdev_rabatel_EPSG32632.tif'
if not os.path.exists(os.path.join(path2data_raw, fname)):
    print(f'Saving {fname} to raw data directory...')
    da_stdev_rabatel.rio.to_raster(os.path.join(path2data_raw, fname))
else: print(f"File {fname} already exists in raw data directory. Skipping save.")

fname = f'gepatsch_velo-flags_rabatel_EPSG32632.tif'
if not os.path.exists(os.path.join(path2data_raw, fname)):
    print(f'Saving {fname} to raw data directory...')
    da_flags_rabatel.rio.to_raster(os.path.join(path2data_raw, fname))
else: print(f"File {fname} already exists in raw data directory. Skipping save.")


#%% CLEANING: 

'''## RAW/CLEAN:
- decompose VELO and DIR to VX and VY
- reprojected velocity (and flags, stdev, etc.) to 25832
- keep flag / high velo blobs
- keep annual velocity files (if they look good)'''

velolist_rabatel = sorted(glob.glob(os.path.join(path2data_raw, 'gepatsch_v_*_rabatel_EPSG32632.tif')))
ds_v_rabatel = xr.open_mfdataset(velolist_rabatel,
                                 combine='nested',
                                 concat_dim='band',
                                 ) ## rabatel velocity is in EPSG:32632
## rename 'band' coord to 'time' and assign time coordinate as the first year of the velocity period
da_v_rabatel = ds_v_rabatel.rename({'band':'time'}).assign_coords(time=[2015, 2016, 2017, 2018, 2019, 2020])['band_data']

'''## decompose VX and VY using the direction
NB: DIR = Flow direction represented as the angle between the flow vector and x-axis. (0°: North))
--> means assuming 
0°   = north
90°  = east
180° = south
270° = west
and 
vx = speed * np.sin(theta)  # eastward / x component
vy = speed * np.cos(theta)  # northward / y component

(rather than default where east=0, and vx = np.cos(theta), vy = np.sin(theta))
'''

vx_rabatel = da_v_rabatel * np.sin(np.deg2rad(da_dir_rabatel))  # eastward / x component
vy_rabatel = da_v_rabatel * np.cos(np.deg2rad(da_dir_rabatel))  # northward / y component


'''## CHECK velocity arrows with dir plot velocity direction as arrows
'''
fig,axs=plt.subplots(1,2, figsize=(16,8))
gdf_2017.boundary.plot(ax=axs[0], edgecolor='steelblue', linewidth=1.5, alpha=0.5, label='Glacier outline 2017')
gdf_2017.boundary.plot(ax=axs[1], edgecolor='steelblue', linewidth=1.5, alpha=0.5, label='Glacier outline 2017')

ax=axs[0]
## make quiver plot of velocity direction
ax.quiver(da_dir_rabatel.x, da_dir_rabatel.y,
          np.cos(np.deg2rad(da_dir_rabatel.values)), ## normal: vx = speed * np.cos(angle)
          np.sin(np.deg2rad(da_dir_rabatel.values)), ## normal: vy = speed * np.sin(angle)
        color='k', scale=50, width=0.002, alpha=0.5)
ax.set_title('default math angles: 0°=east, 90°=north, 180°=west, 270°=south')
ax=axs[1]
## make quiver plot of velocity direction
ax.quiver(da_dir_rabatel.x, da_dir_rabatel.y,
        #   np.cos(np.deg2rad(da_dir_rabatel.values)), ## normal: vx = speed * np.cos(angle)
        #   np.sin(np.deg2rad(da_dir_rabatel.values)), ## normal: vy = speed * np.sin(angle)
          np.sin(np.deg2rad(da_dir_rabatel.values)), ## with NORTH=0: vx = speed * np.sin(angle)
          np.cos(np.deg2rad(da_dir_rabatel.values)), ## with NORTH=0: vy = speed * np.cos(theta)  # northward / y component
            color='k', scale=50, width=0.002, alpha=0.5)
ax.set_title('with NORTH=0: 0°=north, 90°=east, 180°=south, 270°=west')
## zoom box 
padding = 500
for ax in axs:
    ax.set_xlim(df_coords['X0'].min() - padding,
                df_coords['X0'].max() + padding)
    ax.set_ylim(df_coords['Y0'].min() - padding,
                df_coords['Y0'].max() + padding)

## CHECK velocity arrows with dir plot velocity direction as arrows
fig,axs=plt.subplots(1,2, figsize=(10,4))
vx_rabatel.isel(time=-1).plot.imshow(ax=axs[0], cmap='PiYG', vmin=-50, vmax=50)
vy_rabatel.isel(time=-1).plot.imshow(ax=axs[1], cmap='PiYG', vmin=-50, vmax=50)

#%% REPROJECT VELOs
''' ##### REPROJECT VELOCITIES 
NB: Rabatel velocity is in EPSG:32632, and target_crs is EPSG:25832 
    Both are versions of UTM Zone 32N, but with different datums (WGS84 vs ETRS89).
    Therefore, we reporoject velocities, although the difference is likely very small.
##########'''

vx_list = []
vy_list = []
for y in vx_rabatel.time.values: ## only 2015-16 and 2016-17 are in dhdt range, so only keep those for CLEAN data
    # -----------------------------
    # Rotate velocity components to new CRS
    # -----------------------------
    vx_rabatel_rot, vy_rabatel_rot = datafuncs.rotate_velocity_components_reprojection(vx_rabatel.sel(time=y), vy_rabatel.sel(time=y), 
                                                                                       src_epsg="32632", dst_epsg="25832")
    
    # -----------------------------
    # Reproject rasters to EPSG:25832
    # -----------------------------
    da_vx_reproj = vx_rabatel_rot.rio.reproject("EPSG:25832")
    da_vy_reproj = vy_rabatel_rot.rio.reproject("EPSG:25832")

    ## regularize grid back to 50 m resolution
    da_dummy_50m_dst = datafuncs.create_regular_dummy_grid(da_vx_reproj, grid_res=50, crs="EPSG:25832", unit='m')

    da_vx_rerpoj = datafuncs.reproject_match_grid(da_dummy_50m_dst, da_vx_reproj, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)
    da_vy_rerpoj = datafuncs.reproject_match_grid(da_dummy_50m_dst, da_vy_reproj, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)

    ## save to CLEAN directory
    year_str = f'{y}-{y+1}'
    if y <= 2016:
        fname = f'gepatsch_vx_{year_str}.tif'
        if not os.path.exists(os.path.join(path2data_clean, fname)):
            print(f'Saving {fname} to clean data directory...')
            da_vx_rerpoj.rio.to_raster(os.path.join(path2data_clean, fname))
        else: print(f"File {fname} already exists in clean data directory. Skipping save.")

        fname = f'gepatsch_vy_{year_str}.tif'
        if not os.path.exists(os.path.join(path2data_clean, fname)):
            print(f'Saving {fname} to clean data directory...')
            da_vy_rerpoj.rio.to_raster(os.path.join(path2data_clean, fname))
        else: print(f"File {fname} already exists in clean data directory. Skipping save.")

        
        # ## for RAW experiment: apply flags to velocity field and save to CLEAN directory
        # fname = f'gepatsch_vx_{year_str}_flags-masked.tif'
        # da_vx_50m_masked = da_vx_rerpoj.where(da_v_flag_50m==1, np.nan) ## keep values where flag=1 meaning valid points
        # if not os.path.exists(os.path.join(path2data_clean, fname)):
        #     print(f'Saving {fname} to clean data directory...')
        #     da_vx_50m_masked.rio.to_raster(os.path.join(path2data_clean, fname))
        # ## for RAW experiment: apply flags to velocity field and save to CLEAN directory
        # fname = f'gepatsch_vy_{year_str}_flags-masked.tif'
        # da_vy_50m_masked = da_vy_rerpoj.where(da_v_flag_50m==1, np.nan) ## keep values where flag=1 meaning valid points
        # if not os.path.exists(os.path.join(path2data_clean, fname)):
        #     print(f'Saving {fname} to clean data directory...')
        #     da_vy_50m_masked.rio.to_raster(os.path.join(path2data_clean, fname))
        

    vx_list.append(da_vx_rerpoj)
    vy_list.append(da_vy_rerpoj)

## TMP: check differecne with 'direct' reprojections 
# da_vx_TMP = datafuncs.reproject_match_grid(da_dummy_50m_dst.expand_dims('time'), vx_rabatel)
# da_vy_TMP = datafuncs.reproject_match_grid(da_dummy_50m_dst.expand_dims('time'), vy_rabatel)
# diff_TMP = da_vx_TMP.isel(time=0) - da_vx_rerpoj
# ## difference min-max is -0.000684, 0.000428 : so, pretty small.

## stdev raster need to be reprojected but not rotated since its scalar fields
da_v_stdev_50m = datafuncs.reproject_match_grid(da_dummy_50m_dst, da_stdev_rabatel, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)
# da_v_flag_reproj = da_flags_rabatel.rio.reproject("EPSG:25832")
da_v_flag_50m = datafuncs.reproject_match_grid(da_dummy_50m_dst, da_flags_rabatel, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)
## for flag: fill nan values with 1 (0 = flagged data, 1 = good data)
da_v_flag_50m = da_v_flag_50m.fillna(1)
assert da_v_stdev_50m.rio.crs == da_v_flag_50m.rio.crs == 'EPSG:25832', "Reprojected stdev and flag rasters are not in EPSG:25832"

## save to CLEAN directory

fname = f'gepatsch_v-stdev_2016-2021.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    print(f'Saving {fname} to clean data directory...')
    da_v_stdev_50m.rio.to_raster(os.path.join(path2data_clean, fname))
else: print(f"File {fname} already exists in clean data directory. Skipping save.")


fname = f'gepatsch_v-flagged_2016-2021.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    print(f'Saving {fname} to clean data directory...')
    da_v_flag_50m.rio.to_raster(os.path.join(path2data_clean, fname))
else: print(f"File {fname} already exists in clean data directory. Skipping save.")

## for RAW experiment: apply flags to velocity field and save to CLEAN directory
fname = f'gepatsch_v-stdev_2016-2021_flags-masked.tif'
da_v_stdev_50m_masked = da_v_stdev_50m.where(da_v_flag_50m==1, np.nan) ## keep values where flag=1 meaning valid points
if not os.path.exists(os.path.join(path2data_clean, fname)):
    print(f'Saving {fname} to clean data directory...')
    da_v_stdev_50m_masked.rio.to_raster(os.path.join(path2data_clean, fname))



#%% 
''' ## velocity towards homogenized field
- calculate mean velo of all years (2015-2020): used to fill gaps (presumably more 'robust' than only using two years)
    - smooth all-year-mean velocity to remove outliers and noise
- calculate MEAN velo of relevant years (2015-16 and 2016-17)
    - mask FLAGGED areas 
    - fill flagged gaps with all-year-mean 
 '''

da_dummy_10m = datafuncs.create_regular_dummy_grid(da_v_flag_50m, grid_res=10, crs=25832, unit='m')

ds_vx = xr.concat(vx_list, dim='time').assign_coords(time=[2015, 2016, 2017, 2018, 2019, 2020]).rename('vx')
ds_vy = xr.concat(vy_list, dim='time').assign_coords(time=[2015, 2016, 2017, 2018, 2019, 2020]).rename('vy')

## plot vx input
# ds_vx.plot.imshow(col='time', col_wrap=3, cmap='PiYG', vmin=-50, vmax=50)

''' ## original mean field, all years: take mean and smooth'''
da_vx_mean = ds_vx.mean(dim='time', skipna=True)
da_vy_mean = ds_vy.mean(dim='time', skipna=True)
## fil small NaN gaps on-glacier
max_gap = 5
da_vx_mean = (da_vx_mean.interpolate_na( dim="x", method="linear", use_coordinate=False, max_gap=max_gap ).interpolate_na( dim="y", method="linear", use_coordinate=False, max_gap=max_gap ))
da_vy_mean = (da_vy_mean.interpolate_na( dim="x", method="linear", use_coordinate=False, max_gap=max_gap ).interpolate_na( dim="y", method="linear", use_coordinate=False, max_gap=max_gap ))
## fill area outside of glacier with 0  (so that when smoothing we don't get nan-values creeping into the glacier area at the boudns)
da_vx_mean = da_vx_mean.fillna(0)
da_vy_mean = da_vy_mean.fillna(0)

## smooth the mean velocity field to remove outliers and noise
da_vx_smooth = da_vx_mean.rolling(x=3, y=3, center=True).mean() # simple rolling mean with window size 3x3
da_vy_smooth = da_vy_mean.rolling(x=3, y=3, center=True).mean()  # simple rolling mean with window size 3x3


''' ## velocity to use: flag areas and fill gaps '''
## FLAGS: set 1 value to nan
da_flag_valid = da_v_flag_50m.where(da_v_flag_50m==1, np.nan).rename('flag')

## make flagged area a bit more generous: reproject to 10 m with bilinear interpolation, then reproject back to 50 m. 
# This will create flagged area edges with values between 0 and 1, which we can use to define a more generous flagged area.
da_flag_valid_10m = datafuncs.reproject_match_grid(da_dummy_10m.expand_dims('time'), da_v_flag_50m, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
da_flag_valid_50m = datafuncs.reproject_match_grid(da_v_flag_50m, da_flag_valid_10m, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
da_flag_valid_50m = np.round(da_flag_valid_50m, decimals=1).rename('flag_valid') ## round flag values to 0.1 decimals

## mask all flagged velocity
# da_vx_masked = ds_vx.where(da_flag_valid==1, np.nan)
da_vx_masked = ds_vx.where(da_flag_valid_50m>0.8, np.nan) ## using the more generous flagged area. flag=1 is a very big area, flag=0.5 yields (almost) same original area. 
da_vy_masked = ds_vy.where(da_flag_valid_50m>0.8, np.nan)
used_mask = da_flag_valid_50m>0.8 ## binary mask

## set-up desired field: mean value of desired time slices 
da_vx_maskedmean = da_vx_masked.sel(time=[2015,2016]).mean(dim='time', skipna=True)
da_vy_maskedmean = da_vy_masked.sel(time=[2015,2016]).mean(dim='time', skipna=True)
## remove isolated pixels (single pixel with valid value surrounded by NaN) in the masked velocity field
da_vx_maskedmean = da_vx_maskedmean.where(da_vx_maskedmean.notnull().rolling(x=3, y=3, center=True).sum() > 2, np.nan)  # keep only pixels that have at least two valid neighbor in a 3x3 window
da_vy_maskedmean = da_vy_maskedmean.where(da_vy_maskedmean.notnull().rolling(x=3, y=3, center=True).sum() > 2, np.nan)  # keep only pixels that have at least two valid neighbor in a 3x3 window

## fill masked areas with smoothed mean velocity field
da_vx_filled = da_vx_maskedmean.combine_first(da_vx_smooth)
da_vy_filled = da_vy_maskedmean.combine_first(da_vy_smooth)


## calculate velocity magnitude of filled field, and compare difference of original velocity magnitude
da_v_original = np.sqrt(ds_vx.sel(time=[2015,2016]).mean(dim='time', skipna=True)**2 + ds_vy.sel(time=[2015,2016]).mean(dim='time', skipna=True)**2).fillna(0)
da_v_filled = np.sqrt(da_vx_filled**2 + da_vy_filled**2)
da_v_diff = da_v_original - da_v_filled

# ## apply a very simple & small smoothing to the filled velocity field to remove any jumpt at the edges of the gap
# ## --> not needed. the DX of vx does not look like it has extreme jumps
# da_vx_filled_smooth = da_vx_filled.rolling(x=2, y=2, center=True).median()

# ## check difference between filled and original field
# da_vx_diff = da_vx_mean - da_vx_filled
# da_vy_diff = da_vy_mean - da_vy_filled

fig,axs=plt.subplots(2,2, figsize=(12,10))
vmin=-100; vmax=100
ax=axs[0,0]
da_vx_mean.plot.imshow(ax=ax, cmap='PiYG', vmin=vmin, vmax=vmax)
ax.set_title('(1) original mean vx')
ax=axs[0,1]
da_vx_smooth.plot.imshow(ax=ax, cmap='PiYG', vmin=vmin, vmax=vmax)
ax.set_title('(2) Smoothed mean vx')
# ax=axs[0,2]
# da_vx_maskedmean.plot.imshow(ax=ax, cmap='PiYG', vmin=vmin, vmax=vmax)
# ax.set_title('masked mean vx')
ax=axs[1,0]
da_vx_filled.plot.imshow(ax=ax, cmap='PiYG', vmin=vmin, vmax=vmax)
ax.set_title('(4) Filled vx')
ax=axs[1,1]
da_vx_maskedmean.plot.imshow(ax=ax, cmap='PiYG', vmin=vmin, vmax=vmax)
ax.set_title('(3) masked mean vx')
# da_vx_filled_smooth.plot.imshow(ax=ax, cmap='PiYG', vmin=vmin, vmax=vmax)
# ax.set_title('Filled vx with smoothing')
# ax=axs[1,2]
# da_vx_diff.plot.imshow(ax=ax, cmap='RdBu', vmin=-5, vmax=5)
# ax.set_title('Difference between original and filled vx')

# fig.savefig('/Users/mizeboud/Documents/Documents_mizeboud/Projects/ContinuIX/WP1_data/figures/gepatsch_velocity/' + \
#                 f'gepatsch_rabatel-velocity_homogenization_steps.jpg', dpi=300, bbox_inches='tight')

fig,axs=plt.subplots(1,3, figsize=(15,5))
vmin=0; vmax=120
ax=axs[0]
da_v_original.plot.imshow(ax=ax, cmap='viridis', vmin=vmin, vmax=vmax)
ax.set_title('Original velocity magnitude \n (mean 2015-16 & 2016-17)')
ax=axs[1]
da_v_filled.plot.imshow(ax=ax, cmap='viridis', vmin=vmin, vmax=vmax)
ax.set_title('masked+filled velocity magnitude')  
ax=axs[2]
da_v_diff.plot.imshow(ax=ax, cmap='RdBu', vmin=-15, vmax=15)
ax.set_title('Difference between original and \n masked+filled velocity magnitude')
# ax=axs[3]
# used_mask.plot.imshow(ax=ax, cmap='gray', vmin=0, vmax=1)
# ax.set_title('Used mask (1=original, 0=filled)')

# fig.savefig('/Users/mizeboud/Documents/Documents_mizeboud/Projects/ContinuIX/WP1_data/figures/gepatsch_velocity/' + \
#                 f'gepatsch_rabatel-velocity_homogenization_diff.jpg', dpi=300, bbox_inches='tight')


# %%
''' #################
 THICKNESS: 5m resolution 
 ###############'''
da_thickness = xr.open_dataset(os.path.join(path2data_raw, 'gepatsch_h_2006.tif')).isel(band=0)['band_data'].drop_vars('band')
assert da_thickness.rio.crs == 'EPSG:25832', "Thickness raster is not in EPSG:25832"
print(da_thickness.rio.resolution())
da_thickness.plot.imshow()

#%%
''' ##################################
Make Elevation bins
--> 50 m binstep
--> based on earliest DEM if multiple available (assuming glacier is retreating, so earliest DEM has highest elevations)
--> do for both CLEAN and HOMOGNEIZED data; so possibly also resampling to different resolution
##################################
'''

target_res = 10 # meter
target_crs = 'EPSG:25832'
da_dummy_target = datafuncs.create_regular_dummy_grid(da_dhdt, grid_res=10, crs=25832, unit='m')


hmin = da_dem_2017_2m.min().item()
hmax = da_dem_2017_2m.max().item()

## for HOMOGENIZED: do not downsample elev-bin dataArray, but do new binning on donwsampled DEM
da_dem_targetres = datafuncs.reproject_match_grid(da_dummy_target, da_dem_2017_2m, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
da_elev_bins_10m, elev_bin_edges_10m = datafuncs.dicretize_elevation_bins(da_dem_targetres,
                                                     hmin=hmin, hmax=hmax,
                                                     binstep=50)

print('--- elev bins ---')
print(f'.. min max DEM: {np.round(hmin,0):.0f} to {np.round(hmax,0):.0f}')
print('.. bin edges: ', elev_bin_edges_10m)

## save to CLEAN directory
fname = 'gepatsch_elev-bins.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    da_elev_bins_10m.rename('elevation_bins').rio.to_raster(os.path.join(path2data_clean, fname))

#%% 

#%%
''' ##################################
Global uncertainties
Thickness: from H profiles, using thk_min and thk_max to infer [thk] uncertainty range. This is provided over all points --> take average as global value
################################## '''

'''# unct_thickness : unkown '''
'''unct_dhdt : unkown '''
'''unct_velo : std from millan''' 


#%%
''' ##################################
HOMOGENIZED DATA
- fill all NaN values with 0
- do something else for DEM?
################################## '''

target_res = 10 # meter
target_crs = 'EPSG:25832'
da_dummy_target = datafuncs.create_regular_dummy_grid(da_dhdt, grid_res=10, crs=25832, unit='m')


'''## OUTLINE TO MASK'''
# burn outline into raster mask
gdf_outline1 = gdf_2006; year1 = 2006
gdf_outline2 = gdf_2017; year2 = 2017

da_outline_mask1 = (da_dummy_target*year1).rio.clip(gdf_outline1.geometry, gdf_outline1.crs, drop=False) # drop=False to keep the same grid and not drop the pixels outside the outline (which will be set to nodata)
da_outline_mask2 = (da_dummy_target*year2).rio.clip(gdf_outline2.geometry, gdf_outline2.crs, drop=False) # drop=False to keep the same grid and not drop the pixels outside the outline (which will be set to nodata)
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


## initial check that all variables have the same CRS, resolution and shape
## TO UPDATE 
da_var_dict = {'BED':da_bedrock.copy().rename('BED'),
                'DEM': da_dem_2017_2m.copy().rename('DEM'),
                'ELEVBINS': da_elev_bins_10m.copy().rename('ELEVBINS'),
                'THK':da_thickness.copy().rename('THK'),
                'DHDT': da_dhdt.copy().rename('DHDT'),
                'VX': da_vx_filled.copy().rename('VX'),
                'VY': da_vy_filled.copy().rename('VY'),
                'ICEMASK': da_outline_mask.copy().rename('ICEMASK'),
                'UNCT_VX': da_v_stdev_50m.copy().rename('UNCT_VX'),
                'UNCT_VY': da_v_stdev_50m.copy().rename('UNCT_VY'),
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
                #    .rename('ICEMASK')
                   .assign_attrs({'long_name':'Glacier Outline Mask',
                                  'units':'year',
                                  'uncertainty':'unknown',
                                  'crs':target_crs,
                                  'timestamp':'2006 and 2017',
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
                                  'uncertainty':'unknown',
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
                                   'uncertainty':'unknown',
                                   'crs':target_crs,
                                   'timestamp':'2006',
                                   'description':'bedrock elevation, as provided.'
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
                                  'uncertainty':'n/a',
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
                                  'uncertainty':'unknown',
                                   'crs':target_crs,
                                   'timestamp':'2006',
                                   'description':'ice thickness interpolated from GPR. Missing/NaN values were filled with 0.',
                                   'nodata': 0})
                .rio.write_crs(target_crs)
                    )
da_dhdt_hmg = (da_var_dict['DHDT'].copy()
               .fillna(0)
            #    .rename('DHDT')
               .assign_attrs({'long_name':'Surface Elevation Change',
                              'units':'m/year',
                                  'uncertainty':'unknown',
                              'crs':target_crs,
                              'timestamp':'2006-2017',
                              'description':'Annual elevation change. Missing/NaN values were filled with 0.',
                              'nodata': 0})
                .rio.write_crs(target_crs)
               )

da_vx_hmg = (da_var_dict['VX'].copy()
             .fillna(0)
            #  .rename('VX')
             .assign_attrs({'long_name': 'Surface ice velocity (x-component)',
                            'units':'m/year',
                            'uncertainty':'provided as grid UNCT_VX',
                            'crs':target_crs,
                            'timestamp':'2015-2017',
                            'description':'Average velocity for the period 2015-2017. Missing/NaN values were filled with 0.',
                            'nodata': 0
                            })
                .rio.write_crs(target_crs)
)

da_vy_hmg = (da_var_dict['VY'].copy()
             .fillna(0)
            #  .rename('VY')
             .assign_attrs({'long_name': 'Surface ice velocity (y-component)',
                            'units':'m/year',
                            'uncertainty':'provided as grid UNCT_VY',
                            'crs':target_crs,
                            'timestamp':'2015-2017',
                            'description':'Average velocity for the period 2015-2017. Missing/NaN values were filled with 0.',
                            'nodata': 0
                            })
                .rio.write_crs(target_crs)
)

da_vx_stdev_hmg = (da_var_dict['UNCT_VX'].copy()
                   .assign_attrs({'long_name':'Surface Ice Velocity Standard Deviation',
                                  'units':'m/year',
                                  'uncertainty':'n/a',
                                  'crs':target_crs,
                                  'timestamp':'2015-2017',
                                  'description': f'Standard deviation of surface ice velocity over 2015-2017. Universal between X or Y component.'
                                  })
                    .rio.write_crs(target_crs)
)
da_vy_stdev_hmg = (da_var_dict['UNCT_VY'].copy()
                   .assign_attrs({'long_name':'Surface Ice Velocity Standard Deviation',
                                  'units':'m/year',
                                  'uncertainty':'n/a',
                                  'crs':target_crs,
                                  'timestamp':'2015-2017',
                                  'description': f'Standard deviation of surface ice velocity over 2015-2017. Universal between X or Y component.'
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
                da_vx_stdev_hmg,
                da_vy_stdev_hmg
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


## encoding settings for compression and data type; same for all variables
comp = {"zlib": True, 
        "complevel": 5,  ## level of compression; higher number = more compression but slower read/write
        "dtype": "float32", ## 7 digits of precision 
        # "_FillValue": np.float32(-999), ## for remaining NaN values
        }
encoding = {var: comp for var in ds_glacier_hmg.data_vars if var != "spatial_ref"}  # Exclude spatial_ref from encoding
encoding["spatial_ref"] = {}  # No compression for spatial_ref

fname_nc = 'gepatschferner_glacier_observations.nc'

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


fname_nc = 'gepatschferner_glacier_observations.nc'

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

    ax=axs[row,col]
    da_plot.plot.imshow(ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, cbar_kwargs={'shrink': 0.7})
    ax.set_title(var)
    col+=1
    if col >= 4:
        col = 0
        row += 1
[ax.set_aspect('equal') for ax in axs.flatten()];
[ax.set_axis_off() for ax in axs.flatten()];

fig.savefig(os.path.join(path2data_homog, 'gepatschferner_netcdf_vars.png'), dpi=300)
# %%


# %%

# %%
