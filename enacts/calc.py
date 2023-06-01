import numpy as np
import pandas as pd
import xarray as xr

# Date Reading functions
def read_zarr_data(zarr_path):
    """Read and return data in zarr format.

    Parameters
    ------
    zarr_path : str
        String of path to zarr folder.
    Returns
    -------
    zarr_data : Dataset
        Data from zarr folder as multidimensional xarray dataset.
    See Also
    --------
    Notes
     -----
    """
    zarr_data = xr.open_zarr(zarr_path)
    return zarr_data


# Growing season functions

def water_balance_step(sm_yesterday, peffective, et, taw):
    return (sm_yesterday + peffective - et).clip(min=0, max=taw)


def water_balance(
    daily_rain,
    et,
    taw,
    sminit,
    reduce=False,
    time_dim="T",
):
    """Calculate soil moisture.

    Estimates soil moisture from:
        Rainfall, evapotranspiration, total available water and intial soil moisture value. 
    Knowing that:
        `soil_moisture`(t) = `soil_moisture`(t-1) + `daily_rain`(t) - `et`(t) 
    With ceiling and floor respectively at `taw` and 0 at each time step.

    Parameters
    ------
    daily_rain : DataArray
        Daily rainfall data.
    et : DataArray
        Evapotranspiration. Can be a single value with no dimensions or axes.
    taw : DataArray
        Total available water. Can be a single value with no dimensions or axes.
    sminit : DataArray
        Soil moisture initialization. If DataArray, must not have `time_dim` dim.
        Can be a single value with no dimensions or axes.
    time_dim : str, optional             
        Time coordinate in `daily_rain` (default `time_dim`="T"). 
    Returns
    -------
    water_balance : Dataset
        `water_balance` dataset with daily `soil_moisture`.
    See Also
    --------
    Notes
    -----
    Examples
    --------
    """
    # Initializing soil_moisture
    soil_moisture = water_balance_step(
        sminit,
        daily_rain.isel({time_dim: 0}).expand_dims(dim=time_dim),
        et,
        taw,
    )
    if not reduce:
        soil_moisture = soil_moisture.broadcast_like(daily_rain[time_dim])
    # Looping on time_dim
    for t in daily_rain[time_dim][1:]:
        sm_t = water_balance_step(
            soil_moisture.sel({time_dim: t - np.timedelta64(1, "D")}, drop=True),
            daily_rain.sel({time_dim: t}).expand_dims(dim=time_dim),
            et,
            taw,
        )
        if reduce:
            soil_moisture = sm_t
        else:
            soil_moisture.loc[{time_dim: t}] = sm_t.squeeze(time_dim, drop=True)
    soil_moisture.attrs = dict(description="Soil Moisture", units="mm")
    water_balance = xr.Dataset().merge(soil_moisture.rename("soil_moisture"))
    return water_balance


def longest_run_length(flagged_data, dim):
    """ Find the length of the longest run of flagged (0/1) data along a dimension.
    
    A run is a series of 1s not interrupted by 0s.
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
        Array of length of longest run along `dim`
        
    See Also
    --------
    
    Notes
    -----
    The longest run is the maximum value of the discrete difference
    of cumulative flags keeping only the unflagged data.
    Because diff needs at least 2 points,
    we need to keep (where) the unflagged and first and last
    with the cumulative value for last and 0 for first.
    Cumulative flags, where kept, need be propagated by bfill
    so that diff returns 0 or the length of runs.
    
    I believe that it works for unevenly spaced `dim`,
    only we then don't know what the units of the result are.
    
    Examples
    --------
    >>> t = pd.date_range(start="2000-05-01", end="2000-05-29", freq="1D")
    >>> values = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
    ... 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0]
    >>> flags = xr.DataArray(values, dims=["T"], coords={"T": t})
    >>> longest_run_length(flags, "T")
    <xarray.DataArray ()>
    array(7.)
    Attributes: description:  Longest Run Length
    
    """
    
    # Special case coord.size = 1
    lrl = flagged_data
    if lrl[dim].size != 1:
        # Points to apply diff to
        unflagged_and_ends = (flagged_data == 0) * 1
        unflagged_and_ends[{dim: [0, -1]}] = 1
    
        lrl = lrl.cumsum(dim=dim).where(unflagged_and_ends, other = np.nan).where(
            # first cumul point must be set to 0
            lambda x: x[dim] != lrl[dim][0], other=0
        ).bfill(dim).diff(dim).max(dim=dim)
    lrl.attrs = dict(description="Longest Run Length")
    return lrl


def following_dry_spell_length(daily_rain, wet_thresh, time_dim="T"):
    """Compute the count of consecutive dry days (or dry spell length) after each day

    Parameters
    ----------
    daily_rain : DataArray
        Array daily rainfall
    wet_thresh : float
        a dry day is a day when `daily_rain` is lesser or equal to `wet_thresh`
    time_dim : str, optional             
        Daily time dimension of `daily_rain` (default `time_dim` = "T").
 
    Returns
    -------
    DataArray
        Array of length of dry spell immediately following each day along `time_dim`
        
    See Also
    --------
    
    Notes
    -----
    Ideally we would want to cumulate count of dry days backwards
    and reset count to 0 each time a wet day occurs.
    But that is hard to do vectorially.
    But we can cumulatively count all dry days backwayds
    then apply an offset. In more details:
    Cumulate dry days backwards to get all dry days after a day;
    Find when to apply new offset (dry days followed by a wet day);
    Assign cumulated dry days there, Nan elsewhere;
    Propagate backwards and the 0s at the tail.
    And that is going to be the offset.
    Apply offset that is correct for all days followed by dry days.
    Eventually reset days followed by wet days to 0.

    Examples
    --------
    >>> t = pd.date_range(start="2000-05-01", end="2000-05-14", freq="1D")
    >>> values = [0.054383, 0., 0., 0.027983, 0., 0., 7.763758, 3.27952, 13.375934, 4.271866, 12.16503, 9.706059, 7.048605,  0.]
    >>> precip = xr.DataArray(values, dims=["T"], coords={"T": t})
    >>> following_dry_spell_length(precip, 1)
    <xarray.DataArray (T: 14)>
    array([5., 4., 3., 2., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.])
    Coordinates:
      * T        (T) datetime64[ns] 2000-05-01 2000-05-02 ... 2000-05-13 2000-05-14
    """

    # Find dry days
    dry_day = ~(daily_rain > wet_thresh) * 1
    # Cumul dry days backwards and shift back to get the count to exclude day of
    count_dry_days_after_today = dry_day.reindex({time_dim: dry_day[time_dim][::-1]}).cumsum(
        dim=time_dim
    ).reindex({time_dim: dry_day[time_dim]}).shift({time_dim: -1})
    # Find where dry day followed by wet day
    dry_to_wet_day = dry_day.diff(time_dim, label="lower").where(lambda x : x == -1, other=0)
    # Record cumul dry days on that day and put nan elsewhere
    dry_days_offset = (count_dry_days_after_today * dry_to_wet_day).where(lambda x : x != 0, other=np.nan)
    # Back fill nans and assign 0 to tailing ones
    dry_days_offset = dry_days_offset.bfill(dim=time_dim).fillna(0)
    # Subtract offset and shifted wet days are 0.
    dry_spell_length = (count_dry_days_after_today + dry_days_offset) * dry_day.shift({time_dim: -1})
    return dry_spell_length


def onset_date(
    daily_rain,
    wet_thresh,
    wet_spell_length,
    wet_spell_thresh,
    min_wet_days,
    dry_spell_length,
    dry_spell_search,
    time_dim="T",
):
    """Calculate onset date.

    Find the first wet spell of `wet_spell_length` days where: 
        Cumulative rain exceeds `wet_spell_thresh`,
        With at least `min_wet_days` count of wet days (greater than `wet_thresh`),
        Not followed by a dry spell of `dry_spell_length` days of dry days (not wet),
        For the following `dry_spell_search` days
    
    Parameters
    ----------
    daily_rain : DataArray
        Array of daily rainfall values.
    wet_thresh : float
        Rainfall threshold to determine wet day if `daily_rain` is greater than `wet_thresh`.
    wet_spell_length : int
        Length in days of running window when `wet_spell_thresh` is to be met to define a wet spell.
    wet_spell_thresh : float
        Threshold of rainfall to be reached during `wet_spell_length`
        window to define a wet spell.
    min_wet_days : int
        Minimum number of wet days in `wet_spell_length` window when it rained at or above
        `wet_spell_thresh` to be considered a wet spell.
    dry_spell_length : int
        Length in days of dry spell that would invalidate a wet spell as onset date
        if found within the `dry_spell_search` days following the `wet_spell_length`-day window
        that met the wet spell criteria.
    dry_spell_search : int
        Length in days to search for a `dry_spell_length`-day dry spell after a wet spell
        is found that would invalidate the wet spell as onset date.
    time_dim : str, optional
        Time coordinate in `daily_rain` (default `time_dim`="T").       
    Returns
    -------
    onset_delta : DataArray[np.timedelta64]
        Difference between first day of `daily_rain` and first day of onset date.
    See Also
    --------
    Notes
    -----
    Examples
    --------
    """
    # Find wet days
    wet_day = daily_rain > wet_thresh

    # Find 1st wet day in wet spells length
    first_wet_day = wet_day * 1
    first_wet_day = (
        first_wet_day.rolling(**{time_dim: wet_spell_length})
        .construct("wsl")
        .argmax("wsl")
    )

    # Find wet spells
    wet_spell = (
        daily_rain.rolling(**{time_dim: wet_spell_length}).sum() >= wet_spell_thresh
    ) & ((wet_day*1).rolling(**{time_dim: wet_spell_length}).sum() >= min_wet_days)

    # Find dry spells following wet spells
    if dry_spell_search == 0:
        dry_spell_ahead = False
    else:
        dry_day = ~wet_day
        dry_spell = (
            (dry_day*1).rolling(**{time_dim: dry_spell_length}).sum() == dry_spell_length
        )
        # Note that rolling assigns to the last position of the wet_spell
        dry_spell_ahead = (
            (dry_spell*1).rolling(**{time_dim: dry_spell_search})
            .sum()
            .shift(**{time_dim: dry_spell_search * -1})
            != 0
        )

    # Create a mask of 1s and nans where onset conditions are met
    onset_mask = (wet_spell & ~dry_spell_ahead) * 1
    onset_mask = onset_mask.where((onset_mask == 1))

    # Find onset date (or rather last day of 1st valid wet spell)
    # Note it doesn't matter to use idxmax or idxmin,
    # it finds the first max thus the first onset date since we have only 1s and nans
    # all nans returns nan
    onset_delta = onset_mask.idxmax(dim=time_dim)
    onset_delta = (
        onset_delta
        # offset relative position of first wet day
        # note it doesn't matter to apply max or min
        # per construction all values are nan but 1
        - (
            wet_spell_length
            - 1
            - first_wet_day.where(first_wet_day[time_dim] == onset_delta).max(
                dim=time_dim
            )
        ).astype("timedelta64[D]")
        # delta from 1st day of time series
        - daily_rain[time_dim][0]
    ).rename("onset_delta")
    return onset_delta


def cess_date_step(cess_yesterday, dry_spell_length, dry_spell_length_thresh):
    """Updates cessation date delta according to today's soil moisture spell length

    A cessation date is found at the first day of the first dry spell.
    Today's cessation date delta is one day further away than yesterday's.

    Parameters
    ----------
    cess_yesterday : DataArray[np.timedelta64]
        Yesterday's distance in days from cessation date.
        Can not have a time dimension of size greater than 1.
    dry_spell_length : DataArray[np.timedelta64]
        Today's length in days of a dry spell.
    dry_spell_length_thresh : int
        Minimun length of a dry spell to declare cessation date
    
    Returns
    -------
        DataArray[np.timedelta64]
        Updated cessation date delta from today.

    Notes
    -----
    """
    return cess_yesterday.where(
        ~np.isnat(cess_yesterday),
        other=(
            np.timedelta64(2, "D") - dry_spell_length.where(
                dry_spell_length >= np.timedelta64(dry_spell_length_thresh, "D")
            )
        )
    ) - np.timedelta64(1, "D")


def cess_date(
    daily_data,
    is_soil_moisture,
    dry_thresh, 
    dry_spell_length_thresh,
    et=None,
    taw=None,
    sminit=None,
    time_dim="T",
):
    """Calculate cessation date.

    Find first day of the first dry spell where: 
        Soil moisture falls below `dry_thresh` for `min_dry_days` days.

    Parameters
    ----------
    daily_data : DataArray
        Array of daily soil moisture or daily rainfall.
    is_soil_moisture : boolean
        True if `daily_data` is soil moisture, False if it is rainfall.
    dry_thresh : float
        Soil moisture threshold to determine a dry day.
    dry_spell_length_thresh : int
        Minimum number of dry days in a row to be considered a dry spell.
    et : DataArray, optional
        Evapotranspiration used by `water_balance` if `daily_data` is rainfall.
        Can be a single value with no dimensions or axes.
    taw : DataArray, optional
        Total available water used by `water_balance` if `daily_data` is rainfall.
        Can be a single value with no dimensions or axes.
    sminit : DataArray, optional
        Soil moisture initialization used by `water_balance` if `daily_data` is rainfall.
        If DataArray, must not have `time_dim` dim.
        Can be a single value with no dimensions or axes.
    time_dim : str, optional
        Time coordinate in `soil_moisture` (default `time_dim`="T"). 
    
    Returns
    -------
    cess_delta : DataArray[np.timedelta64] 
        Difference between first day of `daily_data` 
        and cessation date.
    
    See Also
    --------
    water_balance

    Notes
    -----
    Examples
    --------
    """
    # Initializing
    if is_soil_moisture:
        sm = daily_data.isel({time_dim: 0})
    else:
        if sminit is None or et is None or taw is None:
            raise Exception("sminit, et and taw can not be None")
        else:
            sminit = xr.DataArray(sminit)
            et = xr.DataArray(et)
            taw = xr.DataArray(taw)
            sm = water_balance_step(sminit, daily_data.isel({time_dim: 0}), et, taw)
    spell_length = (sm < dry_thresh).astype("timedelta64[D]")
    cess_delta = xr.DataArray(np.timedelta64("NaT", "D"))
    # Loop
    for t in daily_data[time_dim][1:]:
        if is_soil_moisture:
            sm = daily_data.sel({time_dim: t})
        else:
            sm = water_balance_step(sm, daily_data.sel({time_dim: t}), et, taw)
        dry_day = sm < dry_thresh
        spell_length = (spell_length + dry_day.astype("timedelta64[D]")) * dry_day
        cess_delta = cess_date_step(
            cess_delta,
            spell_length,
            np.timedelta64(dry_spell_length_thresh, "D"),
        )
    # Delta reference (and coordinate) back to first time point of daily_data
    cess_delta = (
        daily_data[time_dim][-1]
        + cess_delta
        - daily_data[time_dim][0].expand_dims(dim=time_dim)
    )
    return cess_delta


# Time functions
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
    # Dataset output
    daily_tobegroupedby_season = xr.merge([daily_data, seasons_starts, seasons_ends])
    return daily_tobegroupedby_season


# Seasonal Functions

def seasonal_onset_date(
    daily_rain,
    search_start_day,
    search_start_month,
    search_days,
    wet_thresh,
    wet_spell_length,
    wet_spell_thresh,
    min_wet_days,
    dry_spell_length,
    dry_spell_search,
    time_dim="T",
):
    """ Compute yearly seasonal onset dates from daily rainfall.

    Compute yearly dates by utilizing groupby function to group data by season 
    and onset_date function to calculate onset date for each year of grouped data.

    Parameters
    ----------
    daily_rain : DataArray
        Array of daily rainfall values.
    search_start_day : int
        The day part (1-31) of the date to start scanning for onset date.
    search_start_month : int
        The month part (1-12) of the date to start scanning for onset date.
    search_days : int
        Number of days from search start date to scan for onset date.
    wet_thresh : float
        Rainfall threshold to determine wet day.
    wet_spell_length : int
        Length in days of running window when `wet_thresh` is to be met to define a wet spell.
    wet_spell_thresh : float
        Threshold of rainfall to be reached during `wet_spell_length` window to define a wet spell. 
    min_wet_days : int
        Minimum number of wet days in `wet_spell_length` window when it rained at or above 
        `wet_spell_thresh` to be considered a wet spell.
    dry_spell_length : int
        Length in days of dry spell that would invalidate a wet spell as onset date 
        if found within the `dry_spell_search` days following the `wet_spell_length`-day window 
        that met the wet spell criteria.      
    dry_spell_search : int
        Length in days to search for a `dry_spell_length`-day dry spell after a wet spell 
        is found that would invalidate the wet spell as onset date. 
    time_dim : str, optional
        Time coordinate in `soil_moisture` (default `time_dim`="T").
    Returns
    -------
    seasonal_onset_date : Dataset
        Dataset containing days since search start date as timedelta,
        and onset date as datetime for each year in `daily_rain`.
    See Also
    --------
    Notes
    -----
    Function reproducing Ingrid onsetDate function
    '<http://iridl.ldeo.columbia.edu/dochelp/Documentation/details/index.html?func=onsetDate>`_
    Examples
    --------
    """
    # Deal with leap year cases
    if search_start_day == 29 and search_start_month == 2:
        search_start_day = 1
        search_start_month = 3

    # Find an acceptable end_day/_month
    first_end_date = sel_day_and_month(
        daily_rain[time_dim], search_start_day, search_start_month
    )[0] + np.timedelta64(
        search_days
        # search_start_day is part of the search
        - 1 + dry_spell_search
        # in case this first season covers a non-leap year 28 Feb
        # so that if leap years involve in the process, we have enough days
        # and if not, then we add 1 more day which should not cause trouble
        # unless that pushes us to a day that is not part of the data
        # that would make the whole season drop -- acceptable?
        + 1,
        "D",
    )

    end_day = first_end_date.dt.day.values

    end_month = first_end_date.dt.month.values

    # Apply daily grouping by season
    grouped_daily_data = daily_tobegroupedby_season(
        daily_rain, search_start_day, search_start_month, end_day, end_month
    )
    # Apply onset_date
    seasonal_data = (
        grouped_daily_data[daily_rain.name]
        .groupby(grouped_daily_data["seasons_starts"])
        .map(
            onset_date,
            wet_thresh=wet_thresh,
            wet_spell_length=wet_spell_length,
            wet_spell_thresh=wet_spell_thresh,
            min_wet_days=min_wet_days,
            dry_spell_length=dry_spell_length,
            dry_spell_search=dry_spell_search,
        )
        # This was not needed when applying sum
        .drop_vars(time_dim)
        .rename({"seasons_starts": time_dim})
    )
    # Get the seasons ends
    seasons_ends = grouped_daily_data["seasons_ends"].rename({"group": time_dim})
    seasonal_onset_date = xr.merge([seasonal_data, seasons_ends])

    # Tip to get dates from timedelta search_start_day
    # seasonal_onset_date = seasonal_onset_date[time_dim]
    # + seasonal_onset_date.onset_delta
    return seasonal_onset_date

def seasonal_cess_date(
    soil_moisture,
    search_start_day,
    search_start_month,
    search_days,
    dry_thresh,
    dry_spell_length_thresh,
    time_dim="T"
):
    """Use daily moisture data to compute yearly seasonal cessation dates.

    Compute yearly cessation dates by utilizing groupby function to group 
    data by season and cessation_date function to calculate cessation date for each year of data.

    Parameters
    ----------
    soil_moisture : DataArray
        Array of soil moisture values.
    search_start_day : int
        The day part (1-31) of the date to start scanning for cessation date.
    search_start_month : int
        The month part (1-12) of the date to start scanning for cessation date.
    search_days : int
        Number of days from search start date to scan for cessation date.
    dry_thresh : float
        Soil moisture threshold to determine dry day if `dry_thresh` is less than `soil_moisture`
    dry_spell_length_thresh : int
        Minimum number of dry days in a row to be considered a dry spell.
    time_dim : str, optional
        Time coordinate in `soil_moisture` (default `time_dim`="T").
    Returns
    -------
    seasonal_cess_date : Dataset
        Dataset containing days since search start date as timedelta, 
        and cessation date as datetime for each year in soil moisture DataArray.
    See Also
    --------
    Notes
    -----
    Examples
    --------
    """ 
    # Deal with leap year cases
    if search_start_day == 29 and search_start_month == 2:
        search_start_day = 1
        search_start_month = 3

    # Find an acceptable end_day/_month
    first_end_date = sel_day_and_month(
        soil_moisture[time_dim], search_start_day, search_start_month
    )[0] + np.timedelta64(
        search_days,
        "D",
    )

    end_day = first_end_date.dt.day.values

    end_month = first_end_date.dt.month.values

    # Apply daily grouping by season
    grouped_daily_data = daily_tobegroupedby_season(
        soil_moisture, search_start_day, search_start_month, end_day, end_month
    )
    # Apply cess_date
    seasonal_data = (
        grouped_daily_data[soil_moisture.name]
        .groupby(grouped_daily_data["seasons_starts"])
        .map(
            cess_date,
            is_soil_moisture=True,
            dry_thresh=dry_thresh,
            dry_spell_length_thresh=dry_spell_length_thresh,
        )
    ).rename("cess_delta")
    # Get the seasons ends
    seasons_ends = grouped_daily_data["seasons_ends"].rename({"group": time_dim})
    seasonal_cess_date = xr.merge([seasonal_data, seasons_ends])

    # Tip to get dates from timedelta search_start_day
    # seasonal_onset_date = seasonal_onset_date[time_dim]
    # + seasonal_onset_date.onset_delta
    return seasonal_cess_date

def seasonal_sum(
    daily_data,
    start_day,
    start_month,
    end_day,
    end_month,
    min_count=None,
    time_dim="T",
):
    """Calculate seasonal totals of daily data in season defined by day-month edges.
       
    Compute totals  by utilizing groupby function to group data by season
    and then sum the data over the time dimension.
     
    Parameters
    ----------
    daily_data : DataArray
        Daily data to be summed.
    start_day : int
        Day of the start date of the season.
    start_month : int
        Month of the start date of the season.
    end_day : int
        Day of the end date of the season.
    end_month : int
        Month of the end date of the season.
    min_count : int, optional
        Minimum number of valid values to perform operation 
        (default `min_count`=None). 
    time_dim : str, optional
        Time coordinate in `daily_data` (default `time_dim`="T").
    Returns
    -------
    summed_seasons: DataArray
        Totaled daily data for each grouped season.
    See Also
    --------
    Notes
    -----
    Examples
    --------
    """
    grouped_daily_data = daily_tobegroupedby_season(
        daily_data, start_day, start_month, end_day, end_month
    )
    seasonal_data = (
        grouped_daily_data[daily_data.name]
        .groupby(grouped_daily_data["seasons_starts"])
        .sum(dim=time_dim, skipna=True, min_count=min_count)
        #        .rename({"seasons_starts": time_dim})
    )
    seasons_ends = grouped_daily_data["seasons_ends"].rename({"group": time_dim})
    summed_seasons = xr.merge([seasonal_data, seasons_ends])
    return summed_seasons


def probExceed(dfMD, search_start):
    """Calculate probability of exceedance.

    Determining the probability of a seasonal event (onset, cessation) falling
    on a day of the rainy season. The dates for which this is calculated is
    determined by the start date and the output dates of the
    onset / cessation date calculation.

    Parameters
    ----------
    dfMD : DataFrame
        Pandas DataFrame where the first column is datetime values for 
        season event (onset / cessation dates). For the calculation as it
        stands all dates must have the same year in datetime value as is
        used in `search_start`.
    search_start : Datetime
        DateTime value representing the start date with an arbitrary year 
        to be used for calculation.
        ex: 2000-`start_cess_month`-`start-cess-day`.
    Returns
    -------
    cumsum : DataFrame
        Includes number of occurances of each date and days since `search_start`
        with each date's probability of exceedance.                            
    See Also
    --------
    Notes
    -----
    Examples
    --------
    """
    columName = dfMD.columns[0]
    Diff = dfMD[columName] - search_start
    Diff_df = Diff.to_frame()
    counts = Diff_df[columName].value_counts()
    countsDF = counts.to_frame().sort_index()
    cumsum = countsDF.cumsum()
    getTime = Diff_df[columName].dt.total_seconds() / (24 * 60 * 60)
    unique = list(set(getTime))
    unique = [x for x in unique if np.isnan(x) == False]
    cumsum["Days"] = unique
    cumsum["probExceed"] = 1 - cumsum[columName] / cumsum[columName][-1]
    return cumsum
