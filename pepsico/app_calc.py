import xarray as xr
import numpy as np


#This is what we should need for the app
#For the map:
#-- read_data for histo and 1 model
#-- seasonal_data for specific years range
#-- seasanal_data.mean(dim="T") - seasonal_histo.mean(dim="T")
#-- unit_conversion
#-- then maybe some other things to beautify map tbd
#
#For the ts:
#-- read_data for all histo and scenarios
#-- select X and Y
#-- seasonal_data
#-- append or figure out what the graph will need as input to plot histo
#followed by different scenarios
#-- unit_conversion
#-- then maybe some other things to beautify map tbd


def read_data(scenario, model, variable, region, unit_convert=False):
    if region == "US-CA":
        xslice = slice(-154, -45)
        yslice = slice(60, 15)
    elif region == "SAMER":
        xslice = slice(-86, -34)
        yslice = slice(16, -60)
    elif region == "SASIA":
        xslice = slice(59, 94)
        yslice = slice(42, 7)
    elif region == "Thailand":
        xslice = slice(85, 115)
        yslice = slice(28, 2)
    data = xr.open_zarr(
        f'/Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global'
        f'/monthly/{scenario}/{model}/zarr/{variable}'
    )[variable].sel(X=xslice, Y=yslice)
    if unit_convert :
        data = unit_conversion(data)
    return data


def seasonal_data(monthly_data, start_month, end_month, start_year=None, end_year=None):

    #NDJ and DJF are considered part of the year of the 1st month
    if ((end_year != None) and (start_month > end_month)):
        end_year = end_year + 1
    #Reduce data size
    if start_year != None :
        start_year = str(start_year)
    if end_year != None :
        end_year = str(end_year)
    monthly_data = monthly_data.sel(T=slice(start_year, end_year))
    #Find edges of seasons
    start_edges = monthly_data["T"].where(
        lambda x: x["T"].dt.month == start_month, drop=True,
    )
    end_edges = monthly_data["T"].where(
        lambda x: x["T"].dt.month == end_month, drop=True,
    )
    #Select data and edges to avoid partial seasons at the edges of the edges
    monthly_data = monthly_data.sel(T=slice(start_edges[0], end_edges[-1]))
    start_edges = start_edges.sel(T=slice(start_edges[0], end_edges[-1]))
    end_edges = (end_edges
        .sel(T=slice(start_edges[0], end_edges[-1]))
        .assign_coords(T=start_edges["T"])
    )
    #Reduce data size to months in seasons of interest
    months_in_season = (
        (monthly_data["T"] >= start_edges.rename({"T": "group"}))
        & (monthly_data["T"] <= end_edges.rename({"T": "group"}))
    ).sum(dim="group")
    monthly_data = monthly_data.where(months_in_season == 1, drop=True)
    #Create groups of months belonging to same season-year
    seasons_groups = (monthly_data["T"].dt.month == start_month).cumsum() - 1
    #and identified by seasons_starts
    seasons_starts = (
        start_edges.rename({"T": "toto"})[seasons_groups]
        .drop_vars("toto")
        .rename("seasons_starts")
    )

    return (monthly_data
        #Seasonal averages
        .groupby(seasons_starts).mean()
        #Use T as standard name for time dim
        .rename({"seasons_starts": "T"})
        #add seasons_starts/-ends as coords
        .assign_coords(seasons_ends=end_edges)
        .assign_coords(seasons_starts=seasons_starts)
    )


def unit_conversion(variable):
    #if precipitation variable, change from kg to mm per day
    if variable.name == 'pr':
        variable *= 86400
        variable.attrs['units'] = 'mm/day' #rename unit to converted
    elif variable.name in ['tas', 'tasmin', 'tasmax']:
        variable -= 273.15 
        variable.attrs['units'] = '˚C'
    elif variable.name == 'prsn':
        #this is really needed by the colorscale ticks
        variable *= 10e6
        variable.attrs['units'] = f'10e-6 {variable.attrs["units"]}'

    return variable


# This is in enacts calc.py
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


def _cumsum_flagged_diff(flagged_data, dim):
    """ Discrete difference of cumulative flags keeping only the unflagged data
    
    Last 0s before a 1 return the count of consecutive 1s after that 0, except for
    the first value that will return 0 if 0, and the number of consecutive 1s if 1,
    and the last value that is dropped.
    The result is expressed in the units of `dim`, that is assumed evenly spaced.
    
    Parameters
    ----------
    flagged_data : DataArray
        Array of flagged data (0s or 1s)
    dim : str
        dimension of `flagged_data` along which to search for runs
        
    Returns
    -------
    DataArray
        Array of lengths of spells of flags along `dim`
        
    See Also
    --------
    count_days_in_spell, length_of_longest_spell, mean_length_of_spell,
    median_length_of_spell
    
    Notes
    -----
    Because diff needs at least 2 points,
    we need to keep (where) the unflagged and first and last
    with the cumulative value for last and 0 for first.
    Cumulative flags, where kept, need be propagated by bfill
    so that diff returns 0 or the length of runs.
    
    I believe that it works for unevenly spaced `dim`,
    only we then don't know what the units of the result are.

    Is meant to be further reduced to for instance find the longest spell, or the
    numbers of days in spells, etc.
    
    Examples
    --------
    >>> t = pd.date_range(start="2000-05-01", end="2000-05-29", freq="1D")
    >>> values = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
    ... 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0]
    >>> flags = xr.DataArray(values, dims=["T"], coords={"T": t})
    >>> _cumsum_flagged_diff(flags, "T")
    <xarray.DataArray ()>
    array(0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0,
    ... 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 2, 0, 0, 0)    
    """
    
    # Special case coord.size = 1
    cfd = flagged_data
    if cfd[dim].size != 1:
        # Points to apply diff to
        unflagged_and_ends = (flagged_data == 0) * 1
        unflagged_and_ends[{dim: [0, -1]}] = 1
    
        cfd = cfd.cumsum(dim=dim).where(unflagged_and_ends, other = np.nan).where(
            # first cumul point must be set to 0
            lambda x: x[dim] != cfd[dim][0], other=0
        ).bfill(dim).diff(dim)
    return cfd

def count_days_in_spells(flagged_data, dim, min_spell_length=1):
    """ Counts number of days in spells.
    
    Parameters
    ----------
    flagged_data : DataArray
        Array of flagged data (0s or 1s)
    dim : str
        dimension of `flagged_data` along which to count
    min_spell_length: int, optional
        minimum length of for a spell to be accounted for
        
    Returns
    -------
    DataArray
        Array of number of days found in spells along `dim`
        
    See Also
    --------
    _cumsum_flagged_diff
    
    Examples
    --------   
    """

    return _cumsum_flagged_diff(flagged_data, dim).where(
        lambda x : x >= min_spell_length
    ).sum(dim=dim)

def length_of_longest_spell(flagged_data, dim):
    """ Length of longest spells.
    
    Parameters
    ----------
    flagged_data : DataArray
        Array of flagged data (0s or 1s)
    dim : str
        dimension of `flagged_data` along which to count
        
    Returns
    -------
    DataArray
        Array of lengths of longest spells along `dim`
        
    See Also
    --------
    _cumsum_flagged_diff
    
    Examples
    --------   
    """

    return _cumsum_flagged_diff(flagged_data, dim).max(dim=dim)

def mean_length_of_spells(flagged_data, dim, min_spell_length=1):
    """ Mean length of spells.
    
    Parameters
    ----------
    flagged_data : DataArray
        Array of flagged data (0s or 1s)
    dim : str
        dimension of `flagged_data` along which to count
    min_spell_length: int, optional
        minimum length of for a spell to be accounted for
        
    Returns
    -------
    DataArray
        Array of mean length of spells along `dim`
        
    See Also
    --------
    _cumsum_flagged_diff
    
    Examples
    --------   
    """

    return _cumsum_flagged_diff(flagged_data, dim).where(
        lambda x : x >= min_spell_length
    ).mean(dim=dim)

def median_length_of_spells(flagged_data, dim, min_spell_length=1):
    """ Median length of spells.
    
    Parameters
    ----------
    flagged_data : DataArray
        Array of flagged data (0s or 1s)
    dim : str
        dimension of `flagged_data` along which to count
    min_spell_length: int, optional
        minimum length of for a spell to be accounted for
        
    Returns
    -------
    DataArray
        Array of median length of spells along `dim`
        
    See Also
    --------
    _cumsum_flagged_diff
    
    Examples
    --------   
    """

    return _cumsum_flagged_diff(flagged_data, dim).where(
        lambda x : x >= min_spell_length
    ).median(dim=dim)


def _accumulate_spells(flagged_data, axis=0, dtype=None, out=None):
    return xr.apply_ufunc(
        np.frompyfunc(lambda x, y: 0 if y == 0 else x + y, 2, 1).accumulate,
        flagged_data, axis, dtype, out
    )


def spells_length(flagged_data, dim):
    # cumuls 1s and resets counts when 0
    # then rolls data by 1 to position total cumuls on 0s of the flagged data
    # except for the last point that becomes first
    spells = _accumulate_spells(
        flagged_data, axis=flagged_data.get_axis_num(dim)
    ).roll(**{dim: 1})
    # masks out where data falg to rid of accumulating values but last of each spell
    spells_only = spells.where(flagged_data == 0)
    # resets first value to what it was as it could have been erased in previous step
    spells_only[dict(**{dim : 0})] = spells.isel({dim : 0})
    # rolls back to get spells length values on last day of spell
    spells = spells_only.roll(**{dim: -1})
    # Turns remaining 0s to NaN
    return spells.where(spells > 0)
