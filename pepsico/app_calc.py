import xarray as xr
import numpy as np
from scipy.stats import norm


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


def read_data(
    scenario, model, variable, region, time_res="monthly", unit_convert=False
):
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
        f'/{time_res}/{scenario}/{model}/zarr/{variable}'
    )[variable].sel(X=xslice, Y=yslice)
    if unit_convert :
        data = unit_conversion(data)
    #Turns out that some models are centered on noon
    if time_res == "daily" :
        if data["T"][0].dt.hour == 12 :
            data = data.assign_coords({"T" : data["T"] - np.timedelta64(12, "h")})
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


def seasonal_wwc(
    labelled_season_data, variable, frost_threshold, wet_threshold
):
    # Boolean variables need the additional where to return NaNs from False/0 to Nans
    # and the sum parameters for entirely NaNs seasons to remain NaNs and not turn to
    # 0. This is necessary because histo and projections scenarios don't cover the
    # full time period and it's not creating fake results as there are no missing
    # data
    if variable == "frost_days":
        data_ds = (
            (labelled_season_data <= frost_threshold)
            .where(~np.isnan(labelled_season_data))
            .groupby(labelled_season_data["seasons_starts"])
            .sum(skipna=True, min_count=1)
        )
        wwc_units = "days"
    if variable == "dry_days":
        data_ds = (
            (labelled_season_data <= wet_threshold)
            .where(~np.isnan(labelled_season_data))
            .groupby(labelled_season_data["seasons_starts"])
            .sum(skipna=True, min_count=1)
        )
        wwc_units = "days"
    if variable in ["mean_Tmax", "mean_Tmin"]:
        data_ds = labelled_season_data.groupby(
            labelled_season_data["seasons_starts"]
        ).mean()
        wwc_units = "˚C"
    # It takes several 10s of minutes to get empirical quantiles maps
    if variable in ["Tmax_90", "Tmin_10"]:
        quantile = 0.1 if variable == "Tmin_10" else 0.9
        data_ds = labelled_season_data.groupby(
            labelled_season_data["seasons_starts"]
        )
        data_std = data_ds.std()
        data_ds = data_ds.mean()
        for var in data_ds.data_vars:
            data_ds[var].data = xr.apply_ufunc(
                norm.ppf, quantile, kwargs={"loc": data_ds[var], "scale": data_std[var]},
            )
        wwc_units = "˚C"
    # This option is also commented out as it didn't work in its present form. Didn't
    # really expect that it would though.
    if variable == "frost_season_length":
        data_ds = (labelled_season_data <= frost_threshold).groupby(
            labelled_season_data["seasons_starts"]
        )
        data_ds = (
            data_ds[-1:0].idxmax()
            - data_ds.idxmax()
            + np.timedelta64(1, "D")
        )
        wwc_units = "days"
    if variable == "wet_days":
        data_ds = (
            (labelled_season_data > wet_threshold)
            .where(~np.isnan(labelled_season_data))
            .groupby(labelled_season_data["seasons_starts"])
            .sum(skipna=True, min_count=1)
        )
        wwc_units = "days"
    # This is all a bit tedious but I didn't figure out another way to keep
    # seasons_ends and renaming time dim T
    # Can revisit later if this code has a future
    data_ds = data_ds.rename({"seasons_starts" : "T"})
    seasons_ends = labelled_season_data["seasons_ends"].rename({"group": "T"})
    data_ds = data_ds.drop_vars(["seasons_ends", "group"]).assign_coords({"seasons_ends" : seasons_ends})
    #
    for var in data_ds.data_vars:
        data_ds[var].attrs["units"] = wwc_units
    return data_ds


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


def daily_tobegroupedby_season(
    daily_data, start_day, start_month, end_day, end_month, time_dim="T"
):
    """Group daily data by season.
    
    Returns dataset ready to be grouped by with the daily data where all days not in season of interest are dropped.

    season_starts:
        An array where the non-dropped days are indexed by the first day of their season.
            -- to use to groupby
    seasons_ends:
        An array with the dates of the end of the seasons.
           Can then apply groupby on daily_data against seasons_starts, and preserving seasons_ends for the record.
    
    If starting day-month is 29-Feb, uses 1-Mar.
    
    If ending day-month is 29-Feb, uses 1-Mar and uses < rather than <=
        That means that the last day included in the season will be 29-Feb in leap years and 28-Feb otherwise.

    Parameters
    -----------
    daily_data : DataArray
        Daily data to be grouped.
    start_day : int
        Day of the start date  of the season.
    start_month : int
        Month of the start date of the season.
    end_day : int
        Day of the end date of the season.
    end_month : int
        Day of the end date of the season.
    time_dim : str, optional
        Time coordinate in `daily_data` (default `time_dim`="T").
    Returns
    -------
    daily_tobegroupedby_season : Dataset
        Daily data grouped by season using season start date. Dataset includes grouped data
        and `season_starts`, `season_ends` as output variables.  
    See Also
    --------
    Notes
    -----
    Examples
    --------
    """
    # Deal with leap year cases
    if start_day == 29 and start_month == 2:
        start_day = 1
        start_month = 3
    # Find seasons edges
    start_edges = sel_day_and_month(daily_data[time_dim], start_day, start_month)
    if end_day == 29 and end_month == 2:
        end_edges = sel_day_and_month(daily_data[time_dim], 1 , 3, offset=-1)
    else:
        end_edges = sel_day_and_month(daily_data[time_dim], end_day, end_month)
    # Drop dates outside very first and very last edges
    #  -- this ensures we get complete seasons with regards to edges, later on
    daily_data = daily_data.sel(**{time_dim: slice(start_edges[0], end_edges[-1])})
    start_edges = start_edges.sel(**{time_dim: slice(start_edges[0], end_edges[-1])})
    end_edges = end_edges.sel(
        **{time_dim: slice(start_edges[0], end_edges[-1])}
    ).assign_coords(**{time_dim: start_edges[time_dim]})
    # Drops daily data not in seasons of interest
    days_in_season = (
        daily_data[time_dim] >= start_edges.rename({time_dim: "group"})
    ) & (daily_data[time_dim] <= end_edges.rename({time_dim: "group"}))
    days_in_season = days_in_season.sum(dim="group")
    daily_data = daily_data.where(days_in_season == 1, drop=True)
    # Creates seasons_starts that will be used for grouping
    # and seasons_ends that is one of the outputs
    seasons_groups = (daily_data[time_dim].dt.day == start_day) & (
        daily_data[time_dim].dt.month == start_month
    )
    seasons_groups = seasons_groups.cumsum() - 1
    seasons_starts = (
        start_edges.rename({time_dim: "toto"})[seasons_groups]
        .drop_vars("toto")
        .rename("seasons_starts")
    )
    seasons_ends = end_edges.rename({time_dim: "group"}).rename("seasons_ends")
    # I actually changed this from enacts by making both seasons_starts/ends cords
    # rather than vars. This seems better following a comment Aaron made on xcdat and
    # because we local data case provides an xr.Dataset of vars to groupby and we
    # don't want seasons_starts/ends to be groupedby
    daily_tobegroupedby_season = daily_data.assign_coords(
        {"seasons_starts" : seasons_starts, "seasons_ends" : seasons_ends}
    )
    return daily_tobegroupedby_season


def sel_day_and_month(daily_dim, day, month, offset=0):
    """Return a subset of `daily_dim` daily time dimension of corresponding
    `day`/`month` + `offset` day(s) for all years.

    The returned time dimension can then be used to select daily DataArrays.
    Offset is convenient to get days prior to a 1st of March,
    that are not identifiable by a common `day` (28, 29).

    Parameters
    ----------
    daily_dim : DataArray[datetime64[ns]]
        A daily time dimension.
    day : int
        day of the `month`.
    month : int
        month of the year.
    offset : int, optional
        number of days to add to `day`/`month` to offset the selection
        (the default is 0, which implies no offset).

    Returns
    -------
    DataArray[datetime64[ns]]
        a subset of `daily_dim` with all and only `day`-`month` points, offset by `offset` days.
    
    See Also
    --------

    Examples
    --------
    >>> t = pd.date_range(start="2000-05-01", end="20002-04-30", freq="1D")
    >>> values = numpy.arrange(t.size)
    >>> toto = xarray.DataArray(numpy.arrange(61), dims=["T"], coords={"T": t})
    >>> sel_day_and_month(toto["T"], 6, 5)
    <xarray.DataArray 'T' (T: 2)>
    array(['2000-05-06T00:00:00.000000000', '2001-05-06T00:00:00.000000000',]
      dtype='datetime64[ns]')
    Coordinates:
        * T        (T) datetime64[ns] 2000-05-06 2001-05-06

    With an offset of -1 day

    >>> t = pd.date_range(start="2000-01-01", end="20002-01-30", freq="1D")
    >>> toto = xarray.DataArray(numpy.arrange(t.size), dims=["T"], coords={"T": t})
    >>> sel_day_and_month(toto["T"], 1, 3, -1)
    <xarray.DataArray 'T' (T: 2)>
    array(['2000-02-29T00:00:00.000000000', '2001-02-28T00:00:00.000000000',]
      dtype='datetime64[ns]')
    Coordinates:
        * T        (T) datetime64[ns] 2000-02-29 2001-02-28
    """
    return daily_dim.where(
        lambda x: ((x - np.timedelta64(offset, "D")).dt.day == day)
            & ((x - np.timedelta64(offset, "D")).dt.month == month),
        drop=True
    )


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
