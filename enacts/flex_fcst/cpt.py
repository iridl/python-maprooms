import glob
import re
from datetime import datetime
import numpy as np
import cptio
import xarray as xr
from pathlib import Path


def read_file(
    data_path,
    filename_pattern,
    start_date,
    lead_time=None,
    target_time=None,
    ):
    """ Reads a single cpt file for a given start and lead into a xr.Dataset.

    Parameters
    ----------
    data_path : str
        String of the path pointing to cpt datasets.
    filename_pattern : str
        String of the filename pattern name for a given variable's data file.
    lead_time : str
         String of the lead time value to be selected for as is represented in the file name.
    start_date : str
        String of the start date to be selected for as is represented in the file name.
    Returns
    -------
    file_selected : xarray Dataset
        Single CPT data file as multidimensional xarray dataset.
    Notes
    -----
    `filename_pattern` should be most common denominator for any group of datasets,
    so that a single file can be selected using only `lead_time` and `start_date`.
    Examples
    --------
    For files which have naming structure such as the example file:
        CFSv2_SubXPRCP_CCAFCST_mu_Apr_Apr-1-2022_wk1.txt
    And where this file's `lead_time` and `start_date`:
        `lead_time` == 'wk1' and `start_date` == 'Apr-1-2022'
    `filename_pattern` == 'CFSv2_SubXPRCP_CCAFCST_mu_Apr_mystartandlead.txt'
    """
    if lead_time is not None:
        pattern = f"{start_date}_{lead_time}"
    else:
        if filename_pattern == "obs_PRCP_SLtarget.tsv":
            pattern = f"{target_time}"
        else:
            pattern = f"{target_time}_{start_date}"
    full_path = f"{data_path}/{filename_pattern}"
    expanded_name = glob.glob(full_path.replace("SLtarget",pattern))
    if len(expanded_name) == 0:
        read_ds = None
    else:
        file_name = expanded_name[0]
        read_ds = cptio.open_cptdataset(file_name)
    return read_ds


def read_cptdataset(
    data_path,
    filename_pattern_mu,
    filename_pattern_var,
    filename_pattern_obs,
    start_date,
    lead_time=None,
    target_time=None,
    filename_pattern_hcst=None,
):
    if lead_time is not None and target_time is not None:
        raise Exception("I am not sure which of leads or targets to use")
    elif lead_time is not None:
        use_leads = lead_time
        use_targets = None
    elif target_time is not None:
        use_leads = None
        use_targets = lead_time
    else:
        raise Exception("One of leads or targets must be not None")
    fcst_mu = cpt.read_file(
        data_path,
        filename_pattern_mu,
        start_date,
        lead_time=use_leads,
        target_time=use_targets,
    )
    if fcst_mu is not None:
        fcst_mu_name = list(fcst_mu.data_vars)[0]
        fcst_mu = fcst_mu[fcst_mu_name]
    fcst_var = cpt.read_file(
        data_path,
        filename_pattern_var,
        start_date,
        lead_time=use_leads,
        target_time=use_targets,
    )
    if fcst_var is not None:
        fcst_var_name = list(fcst_var.data_vars)[0]
        fcst_var = fcst_var[fcst_var_name]
    obs = cpt.read_file(
        data_path,
        filename_pattern_obs,
        start_date,
        lead_time=use_leads,
        target_time=use_targets,
    )
    if obs is not None:
        obs = obs.squeeze()
        obs_name = list(obs.data_vars)[0]
        obs = obs[obs_name]
    if filename_pattern_hcst it not None:
        hcst = cpt.read_file(
            data_path,
            filename_pattern_hcst,
            start_date,
            lead_time=use_leads,
            target_time=use_targets,
        )
        if hcst is not None:
            hcst = hcst.squeeze()
            hcst_name = list(hcst.data_vars)[0]
            hcst = hcst[hcst_name]
    else:
        hcst = None
    return fcst_mu, fcst_var, obs, hcst


def starts_list(
    data_path,
    filename_pattern,
    regex_search_pattern,
    format_in="%b-%d-%Y",
    format_out="%b-%-d-%Y",
):
    """ Get list of all start dates from CPT files.

    Parameters
    ----------
    data_path : str
        String of the path pointing to cpt datasets.
    filename_pattern : str
        String of the filename pattern name for a given variable's data file.
    regex_search_pattern : str
        String representing regular expression search pattern to find dates in file names.
    format_in : str
        String representing dates format found in file names
    format_out : str
        String representing desired output dates format.
    Returns
    -------
    start_dates : list
        List of strings representing all start dates for the data within `data_path`.
    Notes
    -----
    For more information on regex visit: https://docs.python.org/3/library/re.html
    Test your regex code here: https://regexr.com/
    Examples
    --------
    Regex expression "\w{3}-\w{1,2}-\w{4}" matches expressions that are:
    '{word of 3 chars}-{word between 1,2 chars}-{word of 4 chars}'
    will match dates of format 'Apr-4-2022', 'dec-14-2022', etc.
    """
    filename_pattern = filename_pattern.replace("SLtarget", "*")
    files_name_list = glob.glob(f'{data_path}/{filename_pattern}')
    start_dates = []
    for file in files_name_list:
        start_date = re.search(regex_search_pattern, file)
        start_date_dt = datetime.strptime(start_date.group(), format_in)
        start_dates.append(start_date_dt)
    start_dates = sorted(set(start_dates)) #finds unique dates in the case there are files with the same date due to multiple lead times
    start_dates = [i.strftime(format_out) for i in start_dates]
    return start_dates


def read_pycptv2dataset(data_path):
    data_path = Path(data_path)
    children = list(data_path.iterdir())

    if 'obs.nc' in map(lambda x: str(x.name), children):
        fcst_mu, fcst_var, obs = read_pycptv2dataset_single_target(data_path)
    else:
        mu_mslices = []
        var_mslices = []
        obs_slices = []
        for target in children:
            new_mu, new_var, new_obs = read_pycptv2dataset_single_target(target)
            if len(children) > 1:
                L = (((new_mu["Ti"].dt.month - new_mu["S"].dt.month).squeeze() + 12) % 12).values
                new_mu = new_mu.assign_coords({"L": L}).expand_dims(dim="L")
                new_var = new_var.assign_coords({"L": L}).expand_dims(dim="L")
            mu_mslices.append(new_mu)
            var_mslices.append(new_var)
            obs_slices.append(new_obs)
        fcst_mu = xr.combine_by_coords(mu_mslices)["deterministic"]
        fcst_var = xr.combine_by_coords(var_mslices)["prediction_error_variance"]
        obs = xr.concat(obs_slices, "T")
        obs = obs.sortby(obs["T"])
    return fcst_mu, fcst_var, obs 


def read_pycptv2dataset_single_target(data_path):
    mu_slices = []
    var_slices = []
    for mm in (np.arange(12) + 1) :
        monthly_path = Path(data_path) / f'{mm:02}'
        if monthly_path.exists():
            mu_slices.append(open_var(monthly_path, 'MME_deterministic_forecast_*.nc'))
            var_slices.append(open_var(monthly_path, 'MME_forecast_prediction_error_variance_*.nc'))
    fcst_mu = xr.concat(mu_slices, "S")["deterministic"]
    fcst_mu = fcst_mu.sortby(fcst_mu["S"])
    fcst_var = xr.concat(var_slices, "S")["prediction_error_variance"]
    fcst_var = fcst_var.sortby(fcst_var["S"])
    obs = xr.open_dataset(data_path / f"obs.nc")
    obs_name = list(obs.data_vars)[0]
    obs = obs[obs_name]
    return fcst_mu, fcst_var, obs


def open_mfdataset_nodask(filenames):
    return xr.concat((xr.open_dataset(f) for f in filenames), 'T')


def open_var(path, filepattern):
    filenames = path.glob(filepattern)
    slices = (xr.open_dataset(f) for f in filenames)
    ds = xr.concat(slices, 'T').swap_dims(T='S')
    return ds
   
