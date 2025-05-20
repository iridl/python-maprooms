import glob
import numpy as np
import xarray as xr
from pathlib import Path
from . import predictions


def pycptv2_targets_dict(fcst_ds, start_date=None):
    """ Create dictionaries of targets and leads for a given `start_date`

    Parameters
    ----------
    fcst_ds: xr.Dataset
        pycptv2-like xarray.Dataset
    start_date: str("%YYYY-%mm-%dd"), optional
        start date of the forecast. Default is None in which case
        returns dictionary of targets and leads for the last start

    Returns
    -------
    targets, default: dict, float
        targets is a list of dictionaries where keys are label and value and their
        values are respectively a formatted string of the target dates and a number
        of their lead times, for a given start date.

    See Also
    --------
    read_pycptv2dataset, predictions.target_range_formatting
    """
    if start_date is None :
        fcst_ds = fcst_ds.isel(S=[-1])
    else :
        fcst_ds = fcst_ds.sel(S=[start_date])
    if "L" in fcst_ds.dims:        
        targets = [
            {
                "label": predictions.target_range_formatting(
                    fcst_ds['Ti'].squeeze().sel(L=lead).values,
                    fcst_ds['Tf'].squeeze().sel(L=lead).values,
                    "months",
                ),
                "value": lead,
            } for lead in fcst_ds["L"].where(
                ~np.isnat(fcst_ds["T"].squeeze()), drop=True
            ).values
        ]
    else:
        targets = [{
            "label": predictions.target_range_formatting(
                fcst_ds['Ti'].squeeze().values,
                fcst_ds['Tf'].squeeze().values,
                "months"
            ),
            "value": (((
                fcst_ds['Ti'].dt.month - fcst_ds['S'].dt.month
             ) + 12) % 12).data
        }]
    return targets, targets[0]["value"]


def get_fcst(fcst_conf):
    """ Create a forecast and obs dataset expected by maproom from a set of
    pyCPTv2 nc files

    Parameters
    ----------
    fcst_conf: dict
        dictionary indicating pyCPT datasets configuration (see config)

    Returns
    -------
    fcst_ds, obs : xr.Dataset, xr.DataArray
        dataset of forecast mean and variance and their obs

    See Also
    --------
    read_pycptv2dataset
    """
    if "forecast_path" in fcst_conf :
        fcst_ds, obs = read_pycptv2dataset(fcst_conf["forecast_path"])
    else :
        raise Exception("No synthetic case as of yet")
    return fcst_ds, obs


def read_pycptv2dataset(data_path):
    """ Create a xr.Dataset from yearly pyCPT nc files for one or more targets of the
    year, for a forecast mean and variance variables, appending multiple Starts; and
    the corresponding xr.DataArray of the observations time series used to train
    those targets

    Parameters
    ----------
    data_path: str
        path of set of nc files, organized by targets if multiple ones.

    Returns
    -------
    fcst_ds, obs : xr.Dataset, xr.DataArray
        dataset of forecast mean and variance from `data_path` and their obs

    See Also
    --------
    read_pycptv2dataset_single_target

    Notes
    -----
    Whether `data_path` contains multiple targets or not is determined by whether the
    observation obs.nc file(s) is a direct child of `data_path` or under target-wise
    folders that have no restriction on how they are named.
    If multiple targets, `fcst_ds` is expanded in to a square with the L dimension,
    and so are the Ts coordinates that are turned into variables.
    If single target, `fcst_ds` remains 1D against S only and Ts remain coordiantes
    of S.
    `obs` is in any case always appended and sorted into a natural 1D time series
    """
    data_path = Path(data_path)
    children = list(data_path.iterdir())
    if 'obs.nc' in map(lambda x: str(x.name), children):
        fcst_ds, obs = read_pycptv2dataset_single_target(data_path)
    else:
        expand_L = True if (len(children) > 1) else False
        fcst_ds, obs_slices =  read_pycptv2dataset_single_target(
            children[0], expand_L=expand_L
        )
        obs_slices = [obs_slices]
        if len(children) > 1 :
            for target in children[1:] :
                new_fcst_ds, new_obs = read_pycptv2dataset_single_target(
                    target, expand_L=expand_L
                )
                fcst_ds, new_fcst_ds = xr.align(fcst_ds, new_fcst_ds, join="outer")
                fcst_ds = fcst_ds.fillna(new_fcst_ds)
                obs_slices.append(new_obs)
        obs = xr.concat(obs_slices, "T")
        obs = obs.sortby(obs["T"])
    return fcst_ds, obs


def read_pycptv2dataset_single_target(data_path, expand_L=False):
    """ Create a xr.Dataset from yearly pyCPT nc files for a single target of the
    year, for a forecast mean and variance variables, appending multiple Starts; and
    the corresponding xr.DataArray of the observations time series used to train that
    target

    Parameters
    ----------
    data_path: str
        path of set of nc files for a single target, organized under one or more
        Start month of the year folders mm:02
    expand_L: boolean, optional
        Expand xr.Dataset with a lead L dimension. Default is False

    Returns
    -------
    fcst_ds, obs : xr.Dataset, xr.DataArray
        dataset of forecast mean and variance from `data_path` and their obs

    See Also
    --------
    open_var

    Notes
    -----
    To use in read_pycptv2dataset expecting multiple targets, expand_L must be True
    in which case T coordinates become variables and L is expanded to all variables.
    If single target, expand_L muse be False and Ts remain coordinates of S only.
    """
    mu_slices = []
    var_slices = []
    for mm in (np.arange(12) + 1) :
        monthly_path = Path(data_path) / f'{mm:02}'
        if monthly_path.exists():
            mu_slices.append(open_var(
                monthly_path, 'MME_deterministic_forecast_*.nc', expand_L=expand_L
            ))
            var_slices.append(open_var(
                monthly_path,
                'MME_forecast_prediction_error_variance_*.nc',
                expand_L=expand_L,
            ))
    fcst_mu = xr.concat(mu_slices, "S")
    fcst_var = xr.concat(var_slices, "S")
    fcst_ds = fcst_mu.merge(fcst_var)
    fcst_ds = fcst_ds.sortby(fcst_ds["S"])
    obs = xr.open_dataset(data_path / f"obs.nc")
    obs_name = list(obs.data_vars)[0]
    obs = obs[obs_name]
    return fcst_ds, obs


def open_mfdataset_nodask(filenames):
    return xr.concat((xr.open_dataset(f) for f in filenames), 'T')


def open_var(path, filepattern, expand_L=False):
    """ Create a xr.Dataset of yearly pyCPT nc files for a single Start of the year
    and target of the year (or lead), for a single variable

    Parameters
    ----------
    path: pathlib(Path)
        path where nc files are
    filepattern: str
        files name pattern with year replaced by a *
    expand_L: boolean, optional
        Expand xr.Dataset with a lead L dimension. Default is False

    Returns
    -------
    ds : xr.Dataset
        dataset of `filepattern` variable

    Notes
    -----
    To use in read_pycptv2dataset expecting multiple targets, expand_L must be True
    in which case T coordinates become variables and L is expanded to all variables.
    If single target, expand_L muse be False and Ts remain coordinates of S only.
    """
    filenames = path.glob(filepattern)
    slices = (xr.open_dataset(f) for f in filenames)
    ds = xr.concat(slices, 'T').swap_dims(T='S')
    if expand_L :
        L = (((
            ds.isel(S=[0])["Ti"].dt.month - ds.isel(S=[0])["S"].dt.month
        ) + 12) % 12).data
        ds = ds.reset_coords(["T", "Ti", "Tf"]).expand_dims(dim={"L": L})
    return ds
