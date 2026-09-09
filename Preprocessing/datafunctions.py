import numpy as np
import xarray as xr
import rasterio as rio
import rioxarray as rioxr
import warnings

def count_nan_values_in_glacier(da, outline_gdf):
    """
    Check for NaN values in a DataArray within the glacier outline.
    
    Parameters:
    da (xarray.DataArray): The DataArray to check.
    outline_gdf (geopandas.GeoDataFrame): The glacier outline.
    
    Returns:
    int: The count of NaN values within the glacier outline.
    """
    # Fill NaN with a temporary value
    da_tmp = da.where(~np.isnan(da), -999)
    
    # Clip to glacier outline
    da_tmp = da_tmp.rio.clip(outline_gdf.geometry)
    
    # Count the number of temporary values (-999)
    count_invalid = da_tmp.where(da_tmp == -999).count(dim=['x', 'y']).item()
    
    return count_invalid


def reproject_match_grid( ref_img_da, img_da , resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan):
    ''' Match xarray grid of different spatial resolutions.'''
    
    # Expected order: ('time', 'y', 'x')
    dims = img_da.dims
    if 'time' in dims:
        ref_img_da = ref_img_da.transpose('time','y','x') # CRS is alreadyy written .rio.write_crs(3031, inplace=True)
        img_da = img_da.transpose('time','y','x')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning) ## ignore 'rectified to skew grid conversion' warnings
        # -- reproject (even though same crs) and match grid (extent, resolution and projection)
        img_repr_match = img_da.rio.reproject_match(ref_img_da,resampling=resample_method,nodata=nodata_value) # need to specify nodata, otherwise fills with (inf) number 1.79769313e+308

        # advised to update coords to make the coordinates the exact same due to tiny differences in the coordinate values due to floating precision
        img_repr_match = img_repr_match.assign_coords({
            "y": ref_img_da.y,
            "x": ref_img_da.x,
        })
    
    return img_repr_match.transpose(*dims) # transpose dimension order back to original


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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning) ## ignore 'rectified to skew grid conversion' warnings
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


# -----------------------------
# Function to rotate vectors
# -----------------------------
def transform_velocity_components_epsg4326_to_epsg2056(da_vx, da_vy, 
                                                       src_epsg="4326", dst_epsg="2056"):
    """
    Convert velocity components from geographic east/north components
    to EPSG:2056 x/y components.

    velocity can be in m/day or m/year, as long as the spatial unit is in meters

    For each velocity pixel:
    1. Take the pixel center coordinate in lon/lat.
    2. Use the velocity vector to define a small displacement over one year:
        point_1 = lon/lat position
        point_2 = point_1 shifted by ve, vn in metres
    3. Transform both points to EPSG:2056.
        vx_2056 = x2 - x1
        vy_2056 = y2 - y1
    """
    from pyproj import CRS, Transformer, Geod

    src_crs = CRS.from_epsg(src_epsg)
    dst_crs = CRS.from_epsg(dst_epsg)

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    geod = Geod(ellps="WGS84")

    # Get lon/lat coordinate grids
    lon = da_vx["x"].values
    lat = da_vx["y"].values
    lon2d, lat2d = np.meshgrid(lon, lat)

    vx = da_vx.values
    vy = da_vy.values

    # Velocity magnitude and azimuth in geographic coordinates
    speed = np.sqrt(vx**2 + vy**2)

    # pyproj.Geod uses azimuth clockwise from north:
    # eastward vx, northward vy -> azimuth = atan2(east, north)
    azimuth = np.degrees(np.arctan2(vx, vy))

    # Create a second point after moving by the velocity distance. 
    # Speed: spatial unit should be in meters (doesnt matter if its m/day or m/year)
    lon_end, lat_end, _ = geod.fwd(lon2d, lat2d, azimuth, speed)

    # Project start and end points to EPSG:2056
    x0, y0 = transformer.transform(lon2d, lat2d)
    x1, y1 = transformer.transform(lon_end, lat_end)

    # Difference gives velocity components in EPSG:2056
    vx_2056 = x1 - x0
    vy_2056 = y1 - y0

    # Preserve NaNs
    mask = np.isnan(vx) | np.isnan(vy)
    vx_2056[mask] = np.nan
    vy_2056[mask] = np.nan

    da_vx_rot = da_vx.copy(data=vx_2056).rename("vx_reproj")
    da_vy_rot = da_vy.copy(data=vy_2056).rename("vy_reproj")

    da_vx_rot = da_vx_rot.rio.write_crs(src_crs)
    da_vy_rot = da_vy_rot.rio.write_crs(src_crs)

    return da_vx_rot, da_vy_rot


def rotate_velocity_components_reprojection(da_vx, da_vy, 
                                            src_epsg="4326", dst_epsg="2056"):
    from pyproj import CRS, Transformer, Geod

    src_crs = CRS.from_epsg(src_epsg)
    dst_crs = CRS.from_epsg(dst_epsg)

    # Velocity magnitude and azimuth in original coordinates
    vx = da_vx.values
    vy = da_vy.values
    speed = np.sqrt(vx**2 + vy**2)

    ## set up transformer for reprojection
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    if dst_crs.is_geographic:
        raise ValueError(f"Destination CRS {dst_crs} is geographic (lat/lon). Should adapt code to handle this.")

    if src_crs.is_geographic: ## lat lon coords
        # print(f"Source CRS {src_crs} is geographic (lat/lon).")

        
        geod = Geod(ellps="WGS84")

        # Get lon/lat coordinate grids
        lon = da_vx["x"].values
        lat = da_vx["y"].values
        lon2d, lat2d = np.meshgrid(lon, lat)
        
        # pyproj.Geod uses azimuth clockwise from north:
        # eastward vx, northward vy -> azimuth = atan2(east, north)
        azimuth = np.degrees(np.arctan2(vx, vy))

        # Create a second point after moving by the velocity distance. 
        # Speed: spatial unit should be in meters (doesnt matter if its m/day or m/year)
        lon_end, lat_end, _ = geod.fwd(lon2d, lat2d, azimuth, speed)

        # Project start and end points to EPSG:2056
        x0_dst, y0_dst = transformer.transform(lon2d, lat2d)
        x1_dst, y1_dst = transformer.transform(lon_end, lat_end)

    elif src_crs.is_projected: ## projected coords
        # print(f"Source CRS {src_crs} is projected. (meters)")

        # Coordinate grids in source CRS, already metres
        xcoord = da_vx["x"].values
        ycoord = da_vx["y"].values
        x2d, y2d = np.meshgrid(xcoord, ycoord)

        # Starting points transformed to destination CRS
        x0_dst, y0_dst = transformer.transform(x2d, y2d)
        # Ending points in source CRS after one velocity time-unit
        x_end_src = x2d + vx
        y_end_src = y2d + vy
        # Ending points transformed to destination CRS
        x1_dst, y1_dst = transformer.transform(x_end_src, y_end_src)
    

    # Difference gives velocity components in EPSG:2056
    vx_2056 = x1_dst - x0_dst
    vy_2056 = y1_dst - y0_dst

    # Preserve NaNs
    mask = np.isnan(vx) | np.isnan(vy)
    vx_2056[mask] = np.nan
    vy_2056[mask] = np.nan

    da_vx_rot = da_vx.copy(data=vx_2056).rename("vx_rot")
    da_vy_rot = da_vy.copy(data=vy_2056).rename("vy_rot")

    da_vx_rot = da_vx_rot.rio.write_crs(src_crs)
    da_vy_rot = da_vy_rot.rio.write_crs(src_crs)
    
    return da_vx_rot, da_vy_rot


def dicretize_elevation_bins(da_elev, hmin=None, hmax=None, binstep=100):
    ''' Discretize elevation data into bins of specified step size. 
    Parameters:
    - da_elev: xarray DataArray of elevation data
    - hmin: minimum elevation. Will be rounded down to nearest binstep. If not specified, will be taken from da_elev.
    - hmax: maximum elevation. Will be rounded down to nearest binstep. If not specified, will be taken from da_elev.
    - binstep: step size for elevation bins (default: 100 m)
    
    Returns:
    - da_discretized: xarray DataArray with discretized elevation values
    - elev_bins: array of elevation bin edges (left edges). 
                The last value is still left-edge of the last bin, 
                so full captured elevation range extents to bin[-1]+binstep
    '''
    if hmin is None:
        hmin = da_elev.min().item() 
    if hmax is None:
        hmax = da_elev.max().item()
    hmin = np.floor(hmin/ binstep) * binstep
    hmax = np.ceil(hmax / binstep) * binstep

    # create bins from hmin to hmax
    elev_bins = np.arange(hmin, hmax + binstep, step=binstep) 
    # include right edge for last bin, needed for groupby_bins to include the last bin
    elev_bin_edges_inclRight = np.concatenate((elev_bins, [elev_bins[-1] + binstep])) 

    # Apply binning using groupby_bins and calculate mean for each bin
    bin_means_data = da_elev.groupby_bins(
        group=da_elev,
        bins=elev_bin_edges_inclRight,
        right=True,
        include_lowest=True
    ).mean().values

    # Get bin index for each pixel in the dataArray
    da_elev_bin_idx = xr.apply_ufunc(
        np.digitize,
        da_elev,
        elev_bins,
        kwargs={'right': True}
    )

    # Replace each bin index with its corresponding bin edge value
    da_discretized = xr.apply_ufunc(
        np.vectorize(lambda idx: elev_bin_edges_inclRight[idx]),
        da_elev_bin_idx
    ).where(~np.isnan(da_elev)) # retain NaN values where original data was NaN

    ## add attributes
    attrs = {'description': f'Discretized elevation values into bins of {binstep} m. Using lowest (left-edge) value for each bin.',
            'long_name': 'Elevation bin location',
            'name': 'elevation_bins',
             'bin_step': binstep,
             'bin_left_edges': elev_bins.tolist(),
             'units': 'm'}
    da_discretized.attrs = attrs

    return da_discretized, elev_bins