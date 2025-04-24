import glob
import re
from datetime import datetime
import numpy as np
import cptio
import xarray as xr
from pathlib import Path
from . import predictions


def get_fcst_SandL(fcst_conf):
    """ Returns lists of starts and leads/targets

    Parameters
    ----------
    fcst_conf: dict
        the forecast_path of the dataset (see Notes for tsv case)
    
    Returns
    -------
    start_dates, lead_times: [str("%b-%-d-%Y")], list
        lists of starts and leads/targets

    See Also
    --------
    starts_list, read_pycptv2dataset

    Notes
    -----
    Mostly there to accommodate initialization of tsv case that needs to get S and
    L/target information from the config file, while pyCPTv2 files can be figured out
    through direct reading
    """
    if "forecast_mu_file_pattern" in fcst_conf :
        start_dates = starts_list(
            fcst_conf["forecast_path"],
            fcst_conf["forecast_mu_file_pattern"],
            fcst_conf["start_regex"],
            format_in=fcst_conf["start_format_in"],
            format_out=fcst_conf["start_format_out"],
        )
        if fcst_conf["leads"] is not None :
            lead_times = fcst_conf["leads"]
        else :
            lead_times = fcst_conf["targets"]
    elif "forecast_path" in fcst_conf :
        fcst_ds, _ = read_pycptv2dataset(fcst_conf["forecast_path"])
        start_dates = fcst_ds["S"].dt.strftime("%b-%-d-%Y").values
        if "L" in fcst_ds.dims :
            lead_times = fcst_ds["L"].values 
        else :
            lead_times = None
    else :
        raise Exception("No synthetic case as of yet")
    return start_dates, lead_times


def get_targets_dict(fcst_conf, start_date=None):
    """ Create dictionaries of targets and leads for a given `start_date`

    Parameters
    ----------
    fcst_conf: dict
        the forecast_path of the dataset (see Notes for tsv case)
    start_date: str("%YYYY-%mm-%dd"), optional
        start date of the forecast. Default is None in which case
        returns dictionary of targets and leads for the last start

    Returns
    -------
    targets, default, is_target: dict, float, boolean
        targets is a list of dictionaries where keys are label and value and their
        values are respectively a formatted string of the target dates and a number
        of their lead times, for a given start date.
        is_target applies only to tsv case for which `fcst_conf` can inform on
        targets/leads as targets or as leads

    See Also
    --------
    pycptv2_targets_dict, read_cpt_fcst_leads

    Notes
    -----
    While pyCPTv2 datasets allow to figure all S/L/target just from the content of
    path of the data files, tsv case needs those informed in the config file.
    """
    if "forecast_mu_file_pattern" in fcst_conf :
        targets, default_choice, is_target = read_cpt_fcst_leads(
            fcst_conf, start_date
        )
    else:
        targets, default_choice = pycptv2_targets_dict(
            fcst_conf, start_date=start_date
        )
        is_target = None
    return targets, default_choice, is_target


def pycptv2_targets_dict(fcst_conf, start_date=None):
    """ Create dictionaries of targets and leads for a given `start_date`

    Parameters
    ----------
    fcst_conf: dict
        the forecast_path of the dataset
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
    fcst_ds, _ = read_pycptv2dataset(fcst_conf["forecast_path"])
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


def read_cpt_fcst_leads(fcst_conf, start_date):
    """ Returns dictionary of CPT (tsv case only) targets for a given `start_date`
    """
    if fcst_conf["leads"] is not None and fcst_conf["targets"] is not None:
        raise Exception("I am not sure which of leads or targets to use")
    elif fcst_conf["leads"] is not None:
        leads_values = list(fcst_conf["leads"].values())
        leads_keys = list(fcst_conf["leads"])
        default_choice = list(fcst_conf["leads"])[0]
        is_target = False
    elif fcst_conf["targets"] is not None:
        leads_values = fcst_conf["targets"]
        leads_keys = leads_values
        is_target = True
    else:
        raise Exception("One of leads or targets must be not None")
    start_date = pd.to_datetime(start_date)
    lead_times = {}
    for idx, lead in enumerate(leads_keys):
        if not is_target:
            target_range = predictions.target_range_format(
                leads_values[idx],
                start_date,
                fcst_conf["target_period_length"],
                fcst_conf["time_units"],
            )
        else :
            target_range = leads_values[idx]
        lead_times.update({lead:target_range})
    return lead_times, default_choice, is_target


def get_fcst(fcst_conf, lead_time=None, start_date=None, y_transform=False):
    """ Create a forecast and obs dataset expected by maproom from either a set of
    pyCPTv2 nc files or a set of CPT tsv files

    Parameters
    ----------
    fcst_conf: dict
        dictionary indicating py/CPT datasets configuration (see config)
    lead_time: str, optional
        optional lead to select the dataset against. Default is None in which case
        returns all leads (see specific case of tsv files in Notes)
    start_date: str("%YYYY-%mm-%dd"), optional
        optional start to select the dataset against. Default is None in which case
        returns all starts (see specific case of tsv files in Notes)
    y_transform: boolean, optional
        Set to True if forecast was generated with a y-transform, in which case
        hindcast is read too (only applies to tsv case)

    Returns
    -------
    fcst_ds, obs : xr.Dataset, xr.DataArray
        dataset of forecast mean and variance and their obs

    See Also
    --------
    read_pycptv2dataset, read_cptdataset, read_cpt_fcst_leads, get_fcst_starts

    Notes
    -----
    read_cptdataset (tsv case) reads only one S/L/target at a time which are set to
    last start and first lead/target if not specified. read_cptdataset can (and must)
    take lead_time as a lead (int) or a target (str) which is why lead_time is a str
    which is also what type the callbacks get as Input. Finally, read_cptdataset
    covers the y_transform case.
    read_pycptv2dataset needs not to know beforehand S/L/target to read the data
    files and thus here can return them all. However it never happens in the app and
    there is no state of the app with multiple S/L/target except for the S and target
    controls, that have their own specific get functions. At the moment, there are no
    cases of pyCPTv2 datasets with a y_transform, so it's not covered here and
    hopefully if the case arise, it can be covered automatically by reading the
    files.
    """
    if "forecast_mu_file_pattern" in fcst_conf :
        _, default_choice, is_target = read_cpt_fcst_leads(
            fcst_conf, start_date=start_date
        )
        start_dates = get_fcst_starts(fcst_conf)
        if lead_time is None:
            lead_time = default_choice
        if start_date is None:
            start_date =start_dates[0]
        deterministic, prediction_error_variance, obs, hcst = read_cptdataset(
            lead_time, is_target, start_date, fcst_conf, y_transform=y_transform
        )
        fcst_ds = (
            deterministic.to_dataset()
            .merge(prediction_error_variance)
            .merge(hcst)
        )
    elif "forecast_path" in fcst_conf :
        fcst_ds, obs = read_pycptv2dataset(fcst_conf["forecast_path"])
        if start_date is not None:
            fcst_ds = fcst_ds.sel(S=start_date)
        if lead_time is not None:
            if "L" in fcst_ds.dims:
                fcst_ds = fcst_ds.sel(L=int(lead_time))
        if fcst_ds["S"].size == 1 :
            if "L" in fcst_ds.dims :
                if fcst_ds["L"].size == 1 :
                    obs = obs.where(
                        obs["T"].dt.month == fcst_ds.squeeze()["T"].dt.month.values,
                        drop=True,
                    )
            else :
                obs = obs.where(
                    obs["T"].dt.month == fcst_ds.squeeze()["T"].dt.month.values,
                    drop=True,
                )
    else :
        raise Exception("No synthetic case as of yet")
    return fcst_ds, obs


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
         String of the lead time value to be selected for as is represented in the
         file name.
    start_date : str
        String of the start date to be selected for as is represented in the file
        name.
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


def read_cptdataset(lead_time, is_target, start_date, fcst_conf, y_transform=False):
    """ Creates mean, variance, observation and hindcasts of a single
    Start/Target/Lead CPT forecast (tsv files)
    """
    target_time = lead_time if is_target else None
    if is_target : lead_time = None
    fcst_mu = read_file(
        fcst_conf["forecast_path"],
        fcst_conf["forecast_mu_file_pattern"],
        start_date,
        lead_time=lead_time,
        target_time=target_time,
    )
    if fcst_mu is not None:
        fcst_mu_name = list(fcst_mu.data_vars)[0]
        fcst_mu = fcst_mu[fcst_mu_name]
    fcst_var = read_file(
        fcst_conf["forecast_path"],
        fcst_conf["forecast_var_file_pattern"],
        start_date,
        lead_time=lead_time,
        target_time=target_time,
    )
    if fcst_var is not None:
        fcst_var_name = list(fcst_var.data_vars)[0]
        fcst_var = fcst_var[fcst_var_name]
    obs = read_file(
        fcst_conf["forecast_path"],
        fcst_conf["obs_file_pattern"],
        start_date,
        lead_time=lead_time,
        target_time=target_time,
    )
    if obs is not None:
        obs = obs.squeeze()
        obs_name = list(obs.data_vars)[0]
        obs = obs[obs_name]
    if y_transform:
        hcst = read_file(
            fcst_conf["forecast_path"],
            fcst_conf["hcst_file_pattern"],
            start_date,
            lead_time=lead_time,
            target_time=target_time,
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
        String representing regular expression search pattern to find dates in file
        names.
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
    #finds unique dates in the case there are files with the same date due to
    # multiple lead times
    start_dates = sorted(set(start_dates))
    start_dates = [i.strftime(format_out) for i in start_dates]
    return start_dates


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
