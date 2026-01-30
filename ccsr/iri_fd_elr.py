from pathlib import Path
import xarray as xr
import datetime
import predictions
from dateutil.relativedelta import relativedelta
import numpy as np


def strftimeb2int(strftimeb):
    """Convert month values to integers (1-12) from strings.
 
    Parameters
    ----------
    strftimeb : str
        String value representing months of year.               
    Returns
    -------
    strftimebint : int
        Integer value corresponding to month.
    See Also
    --------
    Notes
    -----
    Examples
    --------
    """
    strftimeb_all = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    strftimebint = strftimeb_all[strftimeb]
    return strftimebint


def get_elr_S(data_path, var):
    """Returns list of dates of forecast issues.
 
    Parameters
    ----------
    data_path : Path
        Path where data files are located.        
    var : str
        forecast variable as found in files names.       
    Returns
    -------
    Reversely sorted list of issue dates as datetime.
    """
    files_list = data_path.glob(f"forecast_mean_{var}_*.nc")
    return sorted([datetime.datetime(
        int(f.name[-7:-3]),
        strftimeb2int(f.name[-10:-7]),
        1,
    ) for f in files_list], reverse=True)


def read_elr(data_path, var, S=None):
    """Returns the forecast mean, ELR parameters, and climatological ELR parameters
     of a given issue.
 
    Parameters
    ----------
    data_path : Path
        Path where data files are located.        
    var : str
        forecast variable as found in files names.
    S : str, optional
        string identifying the issue date as found in file name, e.g. Jan2026
        If None, returns the latest
    Returns
    -------
    Returns the forecast mean, ELR parameters, and climatological ELR parameters
     of a given issue.
    """
    if S is None:
        S = get_elr_S(data_path, var)[0].strftime('%b%Y')
    file_name_var_date = f"{var}_{S}"
    fcst_mean_ds = xr.open_dataset(
        data_path / f"forecast_mean_{file_name_var_date}.nc"
    )
    fcst_coeff_ds = xr.open_dataset(
        data_path / f"forecast_coeff_{file_name_var_date}.nc"
    )
    fcst_clim_coeff_ds = xr.open_dataset(
        data_path / f"forecast_clim_coeff_{file_name_var_date}.nc"
    )["clim"]
    return fcst_mean_ds, fcst_coeff_ds, fcst_clim_coeff_ds
    

def targets_dict(fcst_ds, S=None):
    """Returns a list of dictionaries of leads and target dates for a given S.
 
    Parameters
    ----------
    fcst_ds : xr.Dataset
        Any ELR forecast dataset with lead coords. 
    S : datetime.datetime
        Forecast issue date
    Returns
    -------
    Returns a list of dictionaries of leads and target dates for a given S.
    """
    leads = fcst_ds["lead"]
    toto = [
        {
            "label": predictions.target_range_formatting(
                S + relativedelta(months=int(lead)),
                S + relativedelta(months=(int(lead) + 2)),
                "months",
            ),
            "value": lead,
        } for lead in leads.values
    ]
    return toto


def ELR_poe(ELR_params, quantity, param_coords, ELR_mean=None):
    """ELR probability of exceedance

    Parameters
    ----------
    ELR_params : xr.Dataset
        Datasets with the ELR parameters Bs
    quantity : xr.DataArray
        physical value to compute probability of exceedance of
    param_coords : str
        name of parameters Bs's coordinate
    ELR_mean : xr.DataArray, optional
        ELR forecast mean. Not needed if calculating poe for clim

    Returns
    -------
    xr.DataArray probability of exceedance as a function of quantity's coords

    Notes
    -----
    Given the ELR_params Bi for i={0, 1, 2}
    Consider F = B0 + B1 * ELR_mean + B2 * quantity
    (where ELR_mean = 0 if clim and B1 doesn't exist)
    Then poe = exp(F) / (1 + exp(F))
    """
    B1 = 0 if ELR_mean is None else ELR_params.isel({param_coords: 1}) * ELR_mean
    F = (
        ELR_params.isel({param_coords: 0})
        + B1
        + ELR_params.isel({param_coords: -1}) * quantity
    )
    return np.exp(F) / (1 + np.exp(F))


def ELR_quantity(ELR_clim_params, p):
    """Climatological physical value for percentile p

    Parameters
    ----------
    ELR_clim_params : xr.DataArray
        The climatolofical ELR parmeters
    p : xr.DataArray
        percentile value

    Returns
    -------
    Climatological physical value for percentile p

    Notes
    -----
    This is basically the inverse of ELR_poe for the clim case.
    """
    return (
        np.log(p / ((1 - p) * np.exp(ELR_clim_params.sel(coeff=1))))
        / ELR_clim_params.sel(coeff=2)
    )


def ELR_pdf(cdf, quantity, dim="percentile"):
    """Probability Density Function from CDF

    Parameters
    ----------
    cdf: xr.DataArray
        CDF as a function of coords dim
    quantity: xr.DataArray
        Physical values of the CDF as a function of coords dim
    dim: str, optional
        Name of the common coords of CDF and quantity
    
    Returns
    -------
    PDF as a function of dim

    Notes
    -----
    PDF is the derivative of CDF as a function of quantity
    """
    return cdf.differentiate(dim) * cdf[dim].diff(dim) / quantity.diff(dim)


def ELR_distribution(quantiles, clim, fcst_ds):
    """Returns the CDF and PDF distributions values of forecast and observations for 
    given quantiles
 
    Parameters
    ----------
    quantiles : xr.DataArray
        Values between 0 and 1 against same coords named percentiles 
    clim : xr.DataArray 
        ELR climatological parameters
    fcst_ds : xr.Dataset 
        forecast mean and ELR parameters 

    Returns
    -------
    Returns the CDF and PDF distributions values of forecast and observations for 
    given quantiles

    Notes
    -----

    """
    obs_ppf = ELR_quantity(clim, quantiles)
    fcst_cdf = ELR_poe(
        fcst_ds["coeff"], obs_ppf, "coff", ELR_mean=fcst_ds["fcst_mean"]
    )
    obs_cdf = ELR_poe(clim, obs_ppf, "coeff")
    # Unfortuntely while the clim ELR params return mm/month,
    # the fcst ones expect total mm
    obs_ppf = 3 * obs_ppf
    fcst_pdf = ELR_pdf(fcst_cdf, obs_ppf)
    obs_pdf = ELR_pdf(obs_cdf, obs_ppf)
    # In the IRI Maprooms, the probabilities are computed for all the models
    # and then averaged
    return fcst_cdf.mean("mod"), obs_cdf, obs_ppf, fcst_pdf.mean("mod"), obs_pdf


def proba(fcst_ds, clim, percentile=None, threshold=None, clipv=None):
    """Calculates forecast probability of exceeding historical percentile 
    or threshold

    Parameters
    ----------
    variable : str
        Percentile or Precipitation/Temperature
    clim : xr.DataArray
        clim ELR parameters
    percentile : real
        historical percentile value to compute poe of
    config : 

    """
    assert (
        (
            ((percentile is None) and (threshold is None))
            or ((percentile is not None) and (threshold is not None))
        ),
        ("Please assign either percentile or threshold"),
    )
    if threshold is None :
        threshold = ELR_quantity(clim, percentile)
        if clipv is not None:
            threshold = threshold.clip(min=clipv)
    fcst_cdf = ELR_poe(
        fcst_ds["coeff"], threshold, "coff", ELR_mean=fcst_ds["fcst_mean"]
    )
    return fcst_cdf.mean("mod")
