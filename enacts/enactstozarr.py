import os
import sys
import numpy as np
import xarray as xr
import datetime as dt
from pathlib import Path
import pingrid
from functools import partial
import calc


CONFIG = pingrid.load_config(os.environ["CONFIG"])
VARIABLE = sys.argv[1] #e.g. precip, tmax, tmin -- check your config 
TIME_RES = sys.argv[2] #e.g. daily, or dekadal -- check in your config
INPUT_PATH = (
    f'{CONFIG["datasets"][TIME_RES]["nc_path"]}'
    f'{CONFIG["datasets"][TIME_RES]["vars"][VARIABLE][0]}'
)
OUTPUT_PATH = (
    (
        f'{CONFIG["datasets"][TIME_RES]["zarr_path"]}'
        f'{CONFIG["datasets"][TIME_RES]["vars"][VARIABLE][0]}'
    ) if CONFIG['datasets'][TIME_RES]['vars'][VARIABLE][1] is None
    else (
        f'{CONFIG["datasets"][TIME_RES]["zarr_path"]}'
        f'{CONFIG["datasets"][TIME_RES]["vars"][VARIABLE][1]}'
    )
)
CHUNKS = CONFIG['datasets'][TIME_RES]['chunks']
ZARR_RESOLUTION = CONFIG['datasets'][TIME_RES]["zarr_resolution"]


def set_up_dims(xda, time_res="daily"):
    """Sets up spatial and temporal dimensions from a set of time-dependent netcdf
    ENACTS files.

    To be used in `preprocess` of `xarray.open_mfdataset` .
    Using some Ingrid naming conventions.

    Parameters
    ----------
    xda : DataArray
        from the list of `paths` from `xarray.open_mfdataset`
    time_res : str, optional
        indicates the time resolution of the set of time-dependent files.
        Default is "daily" and other option is "dekadal"
    
    Returns
    -------
    DataArray of X (longitude), Y (latitude) and T (time, daily or dekadal)

    See Also
    --------
    xarray.open_mfdataset, filename2datetime64
    """    
    return xda.expand_dims(T = [filename2datetime64(
        Path(xda.encoding["source"]), time_res=time_res,
    )]).rename({'Lon': 'X','Lat': 'Y'})


def filename2datetime64(file, time_res="daily"):
    """Return time associated with an ENACTS filename in datetime

    In case of dekadal, returns the first day of the dekad (i.e. 1, 11, or 21)

    Parameters
    ----------
    file : pathlib(Path)
        file to extract date from name
    time_res : str, optional
        indicates the time resolution of the file.
        Default is "daily" and other option is "dekadal"
    
    Returns
    -------
    numpy.datetime64

    See Also
    --------
    set_up_dims, convert
    """
    datestr = file.name.split("_")[2]
    year = int(datestr[0:4])
    month = int(datestr[4:6])
    if time_res == "daily":
        day = int(datestr[6:8])
    elif time_res == "dekadal":
        day = (int(datestr[6:7]) - 1) * 10 + 1
    else:
        raise Exception(
            "time resolution must be 'daily' or 'dekadal' "
        )
    return np.datetime64(dt.datetime(year, month, day))


def regridding(data, resolution):
    """Spatial regridding of `data` to `resolution` .

    Does nothing if current resolution is close (according to numpy) to resolution.

    Parameters
    ----------
    data : DataArray
        data of X and Y to regrid.
    resolution : real
        resolution to regrid to.

    Returns
    -------
    DataArray of `data` regridded to `resolution` .
    """
    if not np.isclose(data['X'][1] - data['X'][0], resolution):
    # TODO this method of regridding is inaccurate because it pretends
    # that (X, Y) define a Euclidian space. In reality, grid cells
    # farther from the equator cover less area and thus should be
    # weighted less heavily. Also, consider using conservative
    # interpolation instead of bilinear, since when going to a much
    # coarser resoution, bilinear discards a lot of information. See [1],
    # and look into xESMF.
    #
    # [1] https://climatedataguide.ucar.edu/climate-data-tools-and-analysis/regridding-overview
        print((
            f"Your data will be regridded."
            f"Refer to function documentation for more information on this."
        ))
        data = data.interp(
            X=np.arange(data.X.min(), data.X.max() + resolution, resolution),
            Y=np.arange(data.Y.min(),data.Y.max() + resolution, resolution),
        )    
    return data


def nc2xr(paths, var_name, time_res="daily", zarr_resolution=None, chunks={}):
    """Open mutiple daily or dekadal ENACTS files as a single dataset.

    Optionally spatially regrids and
    coerces all arrays in this dataset into dask arrays with the given chunks.

    Parameters
    ----------
    paths : str or nested sequence of paths
        Either a string glob in the form "path/to/my/files/*.nc"
        or an explicit list of files to open. Paths can be given as strings
        or as pathlib Paths.
    var_name : str
        name of the ENACTS variable in the nc files
    time_res : str, optional
        indicates the time resolution of the set of files.
        Default is "daily" and other option is "dekadal"
    zarr_resolution : real, optional
        spatial resolution to regrid to.
    chunks : int, tuple of int, "auto" or mapping of hashable to int, optional
        Chunk sizes along each dimension X, Y and T.
    
    Returns
    -------
    Xarray.Dataset containing variable `var_name` and coordinates X, Y and T

    See Also
    --------
    xarray.open_mfdataset, set_up_dims, regridding, xarray.DataArray.chunk
    """
    data = xr.open_mfdataset(
        paths,
        preprocess=partial(set_up_dims, time_res=time_res),
        parallel=False,
    )[var_name]
    if zarr_resolution != None:
        print("attempting regrid")
        data = regridding(data, zarr_resolution)
    return xr.Dataset().merge(data.chunk(chunks=chunks))


def convert(
    input_path,
    output_path,
    var_name,
    time_res="daily",
    zarr_resolution=None,
    chunks={}
):
    """Converts a set of ENACTS files into zarr store.

    Either create a new one or append an existing one

    Parameters
    ----------
    input_path : str
        path where the ENACTS nc files are
    output_path : str
        path where the zarr store is (to append to) or will be (to create).
        To create, (last element of the) path is expected not to exist.
        To append, path is expected to point to a zarr store.
    var_name : str
        name of the ENACTS variable in the nc files
    time_res : str, optional
        indicates the time resolution of the set of files.
        Default is "daily" and other option is "dekadal"
    zarr_resolution : real, optional
        spatial resolution to regrid to.
    chunks : int, tuple of int, "auto" or mapping of hashable to int, optional
        Chunk sizes along each dimension X, Y and T.
        
    Returns
    -------
    output_path : where the zarr store has been written

    See Also
    --------
    calc.read_zarr_data, filename2datetime64, nc2xr, xarray.Dataset.to_zarr
    """
    print(f"converting files for: {time_res} {var_name}")
    netcdf = list(sorted(Path(input_path).glob("*.nc")))
    if Path(output_path).is_dir() :
        current_zarr = calc.read_zarr_data(output_path)
        last_T_zarr = current_zarr["T"][-1]
        T_nc = filename2datetime64(netcdf, time_res=time_res).where(lambda x: x > last_T_zarr.values, drop=True)
        if size(T_nc) == 0 :
            print("There are no new nc time steps")
        else :
            netcdf = netcdf[-1 * size(T_nc):]
            print(f'appending nc to zarr from {last_T_zarr.values} to {T_nc[-1]}')
            nc2xr(
                netcdf,
                var_name,
                time_res=time_res,
                zarr_resolution=zarr_resolution,
                chunks=chunks,
            ).to_zarr(store=output_path, append_dim="T")
    else:
        nc2xr(
            netcdf,
            var_name,
            time_res=time_res,
            zarr_resolution=zarr_resolution,
            chunks=chunks,
        ).to_zarr(store=output_path)
    print(f"conversion for {var_name} complete.")
    return output_path

convert(
    INPUT_PATH,
    OUTPUT_PATH,
    VARIABLE,
    time_res=TIME_RES,
    zarr_resolution=ZARR_RESOLUTION,
    chunks=CHUNKS,
)

