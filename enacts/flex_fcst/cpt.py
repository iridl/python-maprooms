import glob
import re
from datetime import datetime
import numpy as np
import cptio
import xarray as xr

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