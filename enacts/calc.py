import numpy as np
import pandas as pd
import xarray as xr
import datetime
import psycopg2
from psycopg2 import sql
import shapely
from shapely import wkb
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry import Polygon

# Date Reading functions

np.random.seed(123)

def get_data(variable, time_res, ds_conf):
    """ Gets ENACTS data for ENACTS Maprooms, read from files or synthetic

     Parameters
    ----------
    variable : str
        string representing ENACTS variable ("precip", "tmin" or "tmax")
    time_res : str
        "daily" or "dekadal" resolution of the desired variable
    ds_conf : dict
        dictionary indicating ENACTS datasets configuration
        (see config)
    
    Returns
    -------
        `xr.DataArray` of ENACTS `variable` for `time_res` time step
    
    See Also
    --------
    read_enacts, synthesize_enacts
    """
    if ds_conf[time_res] == "FAKE" :
        return synthesize_enacts(variable, time_res, ds_conf["bbox"])
    else:
        return read_enacts(variable, ds_conf[time_res])


def read_enacts(variable, dst_conf):
    """ Read ENACTS data

    Parameters
    ----------
    variable : str
        string representing ENACTS variable ("precip", "tmin" or "tmax")
    dst_conf : dict
        dictionary indicating ENACTS zarr paths for a given time resolution
        (see config)
    
    Returns
    -------
        `xr.DataArray` of ENACTS `variable` for `dst_conf` time step
    
    See Also
    --------
    xr.open_zarr
    """
    data_path = dst_conf['vars'][variable][1]
    if data_path is None:
        data_path = dst_conf['vars'][variable][0]
    zarr_path = f"{dst_conf['zarr_path']}{data_path}"
    var_name = dst_conf['vars'][variable][2]
    xrds = xr.open_zarr(zarr_path)
    return xrds[var_name]    


def synthesize_enacts(variable, time_res, bbox):
    """ Synthetize ENACTS data as `xr.DataArray`

    Parameters
    ----------
    variable : str
        string representing ENACTS variable ("precip", "tmin" or "tmax")
    time_res : str
        "daily" or "dekadal" resolution of the desired variable
    bbox : array
        coordinates of bounding box of spatial domain as [W, S, E, N]
    
    Returns
    -------
        `xr.DataArray` of ENACTS-like `variable` at `time_res` time steps
    """
    # Center mu, amplitude amp of the base sinusoid
    # and amplitude of noisy anomalies to apply to it
    characteristics = {
        "precip": {"mu": -2, "amp": 10, "ano_amp": 5},
        "tmin": {"mu": 27, "amp": 3, "ano_amp": 0.6},
        "tmax": {"mu": 32, "amp": 2, "ano_amp": 0.4},
    }
    T = pd.date_range("1991-01-01", datetime.date.today(), name="T")
     # precip peaks in Apr while temp peaks in Oct
    sinusoid = "cos" if variable == "precip" else "sin"
    annual_cycle = getattr(np, sinusoid)(
        2 * np.pi * (T.dayofyear.values / 365.25 - 0.28)
    ).reshape(T.size, 1, 1)
    base_T = (
        characteristics[variable]["mu"]
        + characteristics[variable]["amp"]
        * annual_cycle
    ) + (
        characteristics[variable]["ano_amp"]
        * np.random.randn(annual_cycle.size, 1, 1)
    )
    if variable == "precip":
        # precip is >0
        # and because of mu and amp,
        # he rainy season is a bit shorter than 1/2 the year
        base_T = np.clip(base_T, a_min=0, a_max=None)
    # Coarse lat, lon dims to preserve some spatial homogeneity
    Y = np.arange(bbox[1], bbox[3], 0.5)
    X = np.arange(bbox[0], bbox[2], 0.5)
    XY_rand = 0.1 * np.random.randn(X.size*Y.size).reshape(1, Y.size, X.size)
    return xr.DataArray(
        data=(base_T + base_T * XY_rand),
        coords={"T": T, "Y": Y, "X": X},
        name=variable,
    ).interp(
        X=np.arange(bbox[0], bbox[2], 0.0375), Y=np.arange(bbox[1], bbox[3], 0.0375)
    )


def get_geom(level, conf):
    """ Form a geometric object from sql query or synthetic

    Parameters
    ----------
    level: int
        level from the enumeration of a suite of administrative boundaries listed in
        `conf` . Synthetic case limited to 0 and 1. 
    conf: dict
        dictionary listing desired administrative shapes and their attributes.

    Returns
    -------
    df : pandas.DataFrame
        a pd.DF with columns "label" (dtype=string),
        "key" (string or int depending on the table),
        and "the_geom" (shapely.Geometry)

    See Also
    --------
        synthesize_geom, sql2geom
    """
    if "bbox" in conf["datasets"] :
        return synthesize_geom(conf["datasets"]["bbox"], level=level)
    else:
        return sql2geom(conf["datasets"]["shapes_adm"][level]["sql"], conf["db"])


def sql2GeoJSON(shapes_sql, db_config):
    """ Form a GeoJSON dict from sql request to a database

    Parameters
    ----------
    shapes_sql: str
        sql request
    db_config: dict
        dictionary with host, port, user and dbname information
    
    Returns
    -------
    features: dict
        dictionary with features as key and GeoJSON of shapes_sql as value

    See Also
    --------
    sql2geom, geom2GeoJSON

    Examples
    --------
    shapes_sql: select id_1 as key, name_1 as label,
        ST_AsBinary(the_geom) as the_geom from sen_adm1
    db_config:
        host: postgres
        port: 5432
        user: ingrid
        dbname: iridb
    """
    return geom2GeoJSON(sql2geom(shapes_sql, db_config))


def geom2GeoJSON(df):
    """ Form a GeoJSON dict from a geometric object

    Parameters
    ----------
    df: geometric object
        shapely geometric object
    
    Returns
    -------
    features: dict
        dictionary with features as key and GeoJSON of `geom` as value

    See Also
    --------
    sql2geom, shapely.MultiPolygon, shapely.geometry.mapping
    """
    df["the_geom"] = df["the_geom"].apply(
        lambda x: x if isinstance(x, MultiPolygon) else MultiPolygon([x])
    )
    shapes = df["the_geom"].apply(shapely.geometry.mapping)
    for i in df.index: #this adds the district layer as a label in the dict
        shapes[i]['label'] = df['label'][i]
    return {"features": shapes}


def sql2geom(shapes_sql, db_config):
    """ Form a geometric object from sql query to a database

    Parameters
    ----------
    shapes_sql: str
        sql query
    db_config: dict
        dictionary with host, port, user and dbname information
    
    Returns
    -------
    df : pandas.DataFrame
        a pd.DF with columns "label" (dtype=string),
        "key" (string or int depending on the table),
        and "the_geom" (shapely.Geometry)

    See Also
    --------
    psycopg2.connect, psycopg2.sql, pandas.read_sql, shapely.wkb,

    Examples
    --------
    shapes_sql: select id_1 as key, name_1 as label,
        ST_AsBinary(the_geom) as the_geom from sen_adm1
    db_config:
        host: postgres
        port: 5432
        user: ingrid
        dbname: iridb
    """
    with psycopg2.connect(**db_config) as conn:
        s = sql.Composed(
            [
                sql.SQL("with g as ("),
                sql.SQL(shapes_sql),
                sql.SQL(
                    """
                    )
                    select
                        g.label, g.key, g.the_geom
                    from g
                    """
                ),
            ]
        )
        df = pd.read_sql(s, conn)
    df["the_geom"] = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))
    return df


def synthesize_geom(bbox, level):
    """ Synthesize a geometric object from a bounding box

    Parameters
    ----------
    bbox : array
        coordinates of bounding box of spatial domain as [W, S, E, N]
    level : int
        0 or 1 to mimick a containing admin level (0) with 1 geometry roughly smaller
        than `bbox` or a contained admin level (1) with 2 geometries partitioning
        level 0
    
    Returns
    -------
    df : pandas.DataFrame
        a pd.DF with columns "label" (dtype=string),
        "key" (string or int depending on the table),
        and "the_geom" (shapely.Geometry)

    See Also
    --------
    shapely.geometry.Polygon

    Notes
    -----
    A level 0 contained into bbox is necessary to test the clipping feature since
    `bbox` is also used to generate the fake data.
    """
    west, south, east, north = bbox
    assert (south + 0.25) <= (north - 0.5), (
        "Please extend latitudinal domain of bbox"
    )
    if east < west :
        assert (west + 0.25) >= (east - 0.5), (
            "Please extend longitudinal domain of bbox"
        )
    else :
        assert (west + 0.25) <= (east - 0.5), (
            "Please extend longitudinal domain of bbox"
        )
    west = west + 0.25
    south = south + 0.25
    east = east - 0.5
    norht = north - 0.5
    if level == 0 :
        df = pd.DataFrame({"label" : ["Guyane"], "key": [0], "the_geom": [Polygon([
            [west, south], [west, north], [east, north], [east, south]
        ])]})
    elif level == 1 : #2 triangles partitioning level-0 box at its SW-NE diagnonal
        df = pd.DataFrame({"label" : ["NW", "SE"], "key": [1, 2],"the_geom": [
            Polygon([[west, south], [west, north], [east, north]]),
            Polygon([[west, south], [east, north], [east, south]]),
        ]})
    else:
        raise Exception("level must be 0 or 1")
    return df


def get_taw(ds_conf):
    """ Get TAW data for ENACTS Maprooms, read from file or synthetic

     Parameters
    ----------
    ds_conf : dict
        dictionary indicating TAW file path configuration
        (see config)
    
    Returns
    -------
        `xr.DataArray` of TAW
    
    See Also
    --------
    read_taw, synthesize_taw
    """
    if ds_conf["taw_file"] == "FAKE" :
        return synthesize_taw(ds_conf["bbox"])
    else:
        # At the moment, it's the only case we have
        # if/when other ways to read taw come up,
        # can reintroduce a more sophisticated read_taw function
        return xr.open_dataarray(ds_conf["taw_file"])


def synthesize_taw(bbox):
    """ Synthesize TAW-like data for ENACTS Maprooms

    Parameters
    ----------
    bbox : array
        coordinates of bounding box of spatial domain as [W, S, E, N]
    
    Returns
    -------
        `xr.DataArray` of TAW
    
    See Also
    --------
    xr.open_dataarray
    """
    Y = np.arange(bbox[1], bbox[3], 0.5)
    X = np.arange(bbox[0], bbox[2], 0.5)
    taw = 90 + 30 * np.random.randn(X.size*Y.size).reshape(Y.size, X.size)
    return xr.DataArray(data=taw, coords={"Y": Y, "X": X}, name="taw").interp(
        X=np.arange(bbox[0], bbox[2], 0.0375), Y=np.arange(bbox[1], bbox[3], 0.0375)
    )


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


def cess_date_from_sm(
    daily_sm,
    dry_thresh, 
    dry_spell_length_thresh,
    time_dim="T",
):
    """Calculate cessation date from daily soil moisture.

    A cessation date is found at the first day of the first dry spell
    where a dry spell is defined as at least `dry_spell_length_thresh` consecutive days
    of soil moisture `daily_sm` less than `dry_thresh` .
    
    Parameters
    ----------
    daily_sm : DataArray
        Array of daily soil moisture values.
    dry_thresh : float
        A dry day is when soil mositure `daily_sm` is lesser than `dry_thresh` .
    dry_spell_length_thresh : int
        A dry spell is at least `dry_spell_length_thresh` dry days.
    time_dim : str, optional
        Time coordinate in `daily_sm` (default `time_dim` ="T").   

    Returns
    -------
    cess_delta : DataArray[np.timedelta64]
        Difference between first day of `daily_sm` and cessation date.

    See Also
    --------
    cess_date_step, cess_date_from_rain
    """
    def sm_func(_, t):
        return daily_sm.sel({time_dim: t})

    return _cess_date(dry_thresh, dry_spell_length_thresh, sm_func, daily_sm[time_dim])


def cess_date_from_rain(
    daily_rain,
    dry_thresh, 
    dry_spell_length_thresh,
    et,
    taw,
    sminit,
    time_dim="T",
):
    """Calculate cessation date from daily rainfall.

    A cessation date is found at the first day of the first dry spell
    where a dry spell is defined as at least `dry_spell_length_thresh` consecutive days
    of soil moisture less than `dry_thresh` .
    Soil moisture is estimated through a simple water balance model
    driven by rainfall `daily_rain` , evapotranspiration `et`, soil capacity `taw`
    and initialized at soil moisture value `sminit` .
    
    Parameters
    ----------
    daily_rain : DataArray
        Array of daily rainfall.
    dry_thresh : float
        A dry day is when soil mositure is lesser than `dry_thresh` .
    dry_spell_length_thresh : int
        A dry spell is at least `dry_spell_length_thresh` dry days.
    et : DataArray
        Evapotranspiration.
    taw : DataArray
        Total available water.
    sminit : DataArray
        Soil moisture initialization. If DataArray, must not have `time_dim` dim.
    time_dim : str, optional
        Time coordinate in `daily_rain` (default `time_dim` ="T").   

    Returns
    -------
    cess_delta : DataArray[np.timedelta64]
        Difference between first day of `daily_rain` and cessation date.

    See Also
    --------
    cess_date_step, water_balance_step, cess_date_from_sm

    Notes
    -----
    Cessation date is typically derived from soil moisture.
    However, soil moisture data is rare, while rainfall data is plenty.
    This function is identical to cess_date_from_sm except that at each time step,
    soil moisture is estimated by water_balance_step.
    The result would be the same as applying water_balance to `daily_rain`
    to estimate soil moisture and use that in cess_date_from_sm.
    Using cess_date_from_rain instead would be more efficient
    if the user wishes not to keep the intermediary result of estimated soil moisture.
    """
    sminit = xr.DataArray(sminit)
    et = xr.DataArray(et)
    taw = xr.DataArray(taw)

    def sm_func(sm, t):
        if sm is None:
            sm = sminit
        return water_balance_step(
            sm, daily_rain.sel({time_dim: t}), et, taw
        )

    return _cess_date(dry_thresh, dry_spell_length_thresh, sm_func, daily_rain[time_dim])


def _cess_date(dry_thresh, dry_spell_length_thresh, sm_func, time_coord):
    spell_length = np.timedelta64(0)
    cess_delta = xr.DataArray(np.timedelta64("NaT", "D"))
    sm = None
    for t in time_coord:
        sm = sm_func(sm, t)
        dry_day = sm < dry_thresh
        spell_length = (spell_length + dry_day.astype("timedelta64[D]")) * dry_day
        cess_delta = cess_date_step(
            cess_delta,
            spell_length,
            np.timedelta64(dry_spell_length_thresh, "D"),
        )
    # Delta reference (and coordinate) back to first time point of daily_data
    cess_delta = (
        time_coord[-1]
        + cess_delta
        - time_coord[0].expand_dims(dim=time_coord.name)
    )
    return cess_delta


# Time functions


def intervals_to_points(intervals, to_point="mid", keep_attrs=True):
    """ Given an xr.DataArray of pd.Interval, return an xr.DataArray of the left,
    mid, or right points of those Intervals.

    Parameters
    ----------
    intervals : xr.DataArray(pd.Interval)
        array of intervals
    to_point : str, optional
        "left", "mid" or "right" point of `intervals`
        default is "mid"
    keep_attrs : boolean, optional
        keep attributes from `intervals` to point array
        default is True

    Returns
    -------
    point_array : xr.DataArray
        array of the left, mid or right points of `intervals`

    See Also
    --------
    pandas.Interval

    Notes
    -----
    Should work for any type of array, not just time.
    xr.groupby_bins against dim renames the Interval dim_bins,
    not sure if xr.groupby does the same,
    and what other Xarray functions return Intervals but, depending,
    could generalize the returned array name
    """
    return xr.DataArray(
        data=[getattr(intervals.values[t], to_point) for t in range(intervals.size)],
        coords={intervals.name : intervals},
        name=( # There might be other automatic cases to cover
            intervals.name.replace("_bins", f'_{to_point}')
            if intervals.name.endswith("_bins")
            else "_".join(intervals.name, f'_{to_point}')
        ),
        attrs=intervals.attrs if keep_attrs else {},
    )
    return data


def replace_intervals_with_points(
    interval_data, interval_dim, to_point="mid", keep_attrs=True
):
    """ Replace a coordinate whose values are pd.Interval with one whose values are
    the left edge, center (mid), or right edge of those intervals.

    Parameters
    ----------
    interval_data : xr.DataArray or xr.Dataset
        data depending on a pd.Interval dimension
    interval_dim : str
        name of pd.Interval dimension to be replaced
    to_point : str, optional
        "left", "mid" or "right" point of `interval_dim` intervals
        default is "mid"
    keep_attrs : boolean, optional
        keep attributes from `interval_dim` to replacing point-wise dimension
        default is True

    Returns
    -------
    point_data : xr.DataArray or xr.Dataset
        of which interval dimension has been replaced by point dimension

    See Also
    --------
    pandas.Interval, intervals_to_points, xarray.assign_coords, xarray.swap_dims
    """
    point_dim = intervals_to_points(
        interval_data[interval_dim], to_point=to_point, keep_attrs=keep_attrs
    )
    return (
        interval_data
        .assign_coords({point_dim.name : (interval_dim, point_dim.data)})
        .swap_dims({interval_dim: point_dim.name})
    )


def groupby_dekads(daily_data, time_dim="T"):
    """ Groups `daily_data` by dekads for grouping operations

    Parameters
    ----------
    daily_data : xr.DataArray or xr.Dataset
        daily data
    time_dim : str, optional
        name of daily time dimenstion, default is "T"

    Returns
    -------
    grouped : xr.core.groupby.DataArrayGroupBy or xr.core.groupby.DataArrayGroupBy
        `daily_data` grouped by dekads

    See Also
    --------
    xarray.groupby_bins, xarray.core.groupby.DataArrayGroupBy,
    xarray.core.groupby.DataArrayGroupBy
    """
    # dekad edges are located at midnight
    dekad_edges = pd.date_range(
        start=daily_data[time_dim][0].dt.floor("D").values,
        end=(daily_data[time_dim][-1] + np.timedelta64(1, "D")).dt.floor("D").values,
        freq="1D",
    )
    dekad_edges = dekad_edges.where(
        (dekad_edges.day == 1) | (dekad_edges.day == 11) | (dekad_edges.day == 21)
    ).dropna()
    assert dekad_edges.size > 1, (
        "daily_data must span at least one full dekad (need 2 edges to form 1 bin)"
    )
    return daily_data.groupby_bins(daily_data[time_dim], dekad_edges, right=False)


def resample_interval_to_daily(time_series, is_intensive=None, time_dim="T_bins"):
    """ Resample any (interval-based) time series to daily

    Parameters
    ----------
    time_series : xr.DataArray or xr.Dataset
        data depending on time intervals greater or equal then a day
    is_intensive : boolean, optional
        indicate the "extensive" or "intensive" property of `time_series` .
        Upsampling to daily requires intensive data.
        If False, make intensive by dividing by length of intervals in days
        Default is None in which case: if units end with "/day", considers intensive,
        else, considers extensive
    time_dim : str, optional
        name of interval time dimenstion, default is "T_bins"

    Returns
    -------
    time_series : xr.DataArray or xr.Dataset
        `time_series` resampled to daily

    See Also
    --------
    pandas.Interval, intervals_to_points, replace_intervals_with_points,
    xr.DataArray.resample, xr.Dataset.resample

    Notes
    -----
    The day is considered the smallest unit or interval of time in the sense that a
    time dimensions expressed as time points is considered equivalent to intervals of
    length 1 day. There may be generalization to make to adapt to the actual smallest
    time unit of this ecosystem which is the ns.
    In thermodynamics (at the core of climate science), quantities can be categorized
    as being intensive or extensive to identify how they change when a system changes
    in size: 2 systems merging add up (extensive) their mass and volume, but they
    don't (intensive) their density (more at
    https://en.wikipedia.org/wiki/Intensive_and_extensive_properties).
    Closer to what we care about here, temperature is intensive so a monthly value
    can be upsampled to daily by assigning same value to all day (implicitely
    admitting that the monthly value is a daily average); but precipitation is
    extensive so that so a monthly value can not be upsampled to daily by simply
    reassigning it (if it rains 300mm in a month, it can rain 300m as well in only
    one day of the month -- and 0 all the other days). However, monthly precipitation
    expressed in mm/day is intensive.
    Extent property could be inferred in more cases, e.g. Kelvin is intensive.
    Waiting for pint to figure that all out.
    This function differ from xr.DataArray/Dataset.resample in that resample expects
    `time_dim` to be datetime-like but it doesn't consider that pd.Interval of
    datetime-like is (probably because it reasons in terms of frequency, ignoring
    width (intervals)).
    """
    if isinstance(time_series[time_dim].values[0], pd._libs.interval.Interval):
        # else time_dim is not intervals thus points thus considered daily already
        # make daily for computations
        if is_intensive is None :
            # There are a lot more cases to cover
            if "units" in time_series.attrs :
                is_intensive = "/day" in time_series.attrs["units"]
            else :
                is_intensive = False
        if not is_intensive : # Can only ffill intensive data
            time_series = time_series / [
                time_series[time_dim].values[t].length.days
                for t in range(time_series[time_dim].size)
            ]
            if "units" in time_series.attrs :
                time_series.attrs["units"] = f'{time_series.attrs["units"]}/day'
        time_dim_left = ( # There might be other automatic cases to cover
            # Same logic as in intervals_to_points
            time_dim.replace("_bins", "_left") if time_dim.endswith("_bins")
            else "_".join(time_dim, "_left")
        )
        time_dim_right = (
            time_dim.replace("_bins", "_right") if time_dim.endswith("_bins")
            else "_".join(time_dim, "right")
        )
        time_series = xr.concat([
            replace_intervals_with_points(time_series, time_dim, to_point="left"),
            # Need to cover entirely the last interval
            replace_intervals_with_points(
                time_series.isel({time_dim : [-1]}), time_dim, to_point="right"
            ).rename({time_dim_right : time_dim_left}),
        ], dim=time_dim_left)
        time_series = (
            time_series.resample({time_dim_left: "1D"}).ffill()
            # once filled, can drop the open right point of last interval
            .isel({time_dim_left : slice(0, -1)})
        )
    return time_series


def regroup(time_series, group="1D", time_dim="T"):
    """ Regroup any type of interval-based time series to another

    Parameters
    ----------
    time_series : xr.DataArray or xr.Dataset
        data depending on time intervals greater or equal then a day
    group: str, int, or array-like[pandas.DatetimeIndex]
        indicates the new type of intervals to regroup to.
        As string, must be: nD, pentad, 8day, dekad, 16day, nM, d-d Mmm
        or d Mmm - d Mmm. See Notes for details.
        As integer or array-like[pandas.DatetimeIndex],
        see xr.DataArray.groupby_bins' `bins` Parameter
    time_dim : str, optional
        name of interval time dimenstion, default is "T"

    Returns
    -------
    regrouped : xr.core.groupby.DataArray/DatasetGroupBy
        `time_series` grouped to specified time intervals groups

    See Also
    --------
    resample_interval_to_daily, pandas.DatetimeIndex, xarray.DataArray.groupby_bins

    Notes
    -----
    The day is considered the smallest unit or interval of time (see Note of
    resample_interval_to_daily). In this implementation, all `time_series` inputs are
    resampled to daily no matter the `group` of interest. It may not be necessary
    and more efficient if `group` is coarser than a day (so nearly all cases). The
    intersection of `time_series` intervals and `group` intervals would form the
    coarsest partition of `time_dim` and weights could be applied depending on
    `method` .
    The seasonal grouping is of different nature than all other groupings. While all
    others make a partition of time, the purpose of the seasonal grouping is to make
    a yearly time series of a tailored sesason (e.g. 19 Jan - 29 Mar). However, the
    case fits. As of now, the selection of the season of interest is to be done after
    applyting `regroup` . It could be incorporated into `regroup` and constitute the
    specificity of this case. Or the case could be removed altogether and put in
    another function.
    Outputs of `regroup` that have a yearly periodicity (ie all except nD, some nM,
    int and some array-like[pandas.DatetimeIndex]), could be used to split the time
    dimension in 2: years and intervals of the year; which in turn could be reduced
    to make climatologies or yearly time series of a given interval.
    Known groups:
    * nD (e.g. 7D): intervals of n (e.g. 7) days from first day of `time_dim`
    * pentad: partition of the year in 5-day intervals. In leap years, Feb 29 is
    included in the 25 Feb - 1 Mar interval, making it 6-day long. E.g. at
    https://iridl.ldeo.columbia.edu/expert/SOURCES/.NOAA/.NCEP/.CPC/.FEWS/.DAILY/.est_prcp/pentadAverage
    * 8day: partition of the year in 8-day intervals. The last interval is used to
    adjust the partitioning and is 26/27-31 Dec depending on leap years. E.g. at
    https://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.1km/.8day/.version_006/.Terra/.NY/.Day/.LST
    * dekad: partition of the months in 3 10-day intervals, except for the last dekad
    of the month that runs until the end of the month (from the 21st -- thus can be
    8, 9, 10 or 11 -day long)
    * 16day: partition of the year in 16-day intervals. The last interval is used to
    adjust the partitioning and is 18/19-31 Dec depending on leap years. E.g. at
    https://iridl.ldeo.columbia.edu/SOURCES/.USGS/.LandDAAC/.MODIS/.version_006/.EAF/.NDVI
    * nM (e.g. 5D): intervals of n (e.g. 5) months from first full month of `time_dim`
    * d-d Mmm or d Mmm - d Mmm (e.g. 21-29 Mar or 19 Jan - 29 Mar): 2 seasons to
    partition time against. The 2nd season is the complentary to the one given to
    `group` (e.g. 30 Mar - 18 Jan).
    * int (e.g. 7): number of equally sized intervals in `time_dim` (see
    xr.DataArray.groupby_bins' `bins` for details).
    * array-like[pandas.DatetimeIndex]: edges of time intervals  (see
    xr.DataArray.groupby_bins' `bins` for details).
    """
    time_series = resample_interval_to_daily(time_series, time_dim=time_dim)
    edges_base = pd.date_range(
        # Flooring needed only if time_series already daily
        start=time_series[time_dim][0].dt.floor("D").values,
        # need one more day since right is excluded
        end=(time_series[time_dim][-1] + np.timedelta64(1, "D")).dt.floor("D").values,
        freq="1D",
    )
    if isinstance(group, str) :
        # Form bins according to group
        if group.endswith("D") :
            bins = np.array(
                [edges_base[t] for t in range(0, edges_base.size, int(group[:-1]))]
            )
        elif group == "pentad" :
            # 29 Feb always in last pentad of Feb
            bins = edges_base.where(
                (edges_base.dayofyear % 5) == (
                    1 - np.array([
                        pd.Timedelta(
                            ((eb.dayofyear > 59) * (not eb.is_leap_year)), "D"
                        ).days for eb in edges_base
                    ])
                )
            ).dropna()
        elif group == "8day" :
            # last period of year used to adjusting
            bins = edges_base.where((edges_base.dayofyear % 8) == 1).dropna()
        elif group == "dekad" :
            bins = edges_base.where(
                (edges_base.day == 1)
                | (edges_base.day == 11)
                | (edges_base.day == 21)
            ).dropna()
        elif group == "16day" :
            # last period of year used to adjusting
            bins = edges_base.where((edges_base.dayofyear % 16) == 1).dropna()
        elif group.endswith("M") :
            bins = edges_base.where(edges_base.day == 1).dropna()
            bins = np.array([bins[t] for t in range(0, bins.size, int(group[:-1]))])
        elif "-" in group :
            # e.g. "29 Feb - 30 Mar" or "2-29 Mar"
            # This case usage is to keep only the season of interest given as input.
            # Thus not to form a partition of time as in other cases and could be
            # moved to another function. Or if kept could include the selection of
            # said season of interest. (Or just leave as is).
            if " - " in group :
                start_day = group.split()[0]
                start_month = group.split()[1]
                end_day = group.split()[3]
                end_month = group.split()[4]
            else:
                start_day = group.split()[0].split("-")[0]
                end_day = group.split()[0].split("-")[1]
                start_month = group.split()[1]
                end_month = start_month
            start_day = int(start_day)
            end_day = int(end_day)
            if start_day == 29 and start_month == "Feb" :
                # don't allow start on 29 Feb: this is pushy
                start_day = 1
                start_month = "Mar"
            offset = 1
            if end_day == 29 and end_month == "Feb" :
                end_day = 1
                end_month = "Mar"
                offset = 0
            bins = edges_base.where(
                (
                    (edges_base.day == start_day)
                    & [
                        (edges_base.month_name()[t][:3] == start_month)
                        for t in range(edges_base.size)
                    ]
                )
                | ( # group end is inclusive
                    ((edges_base - pd.Timedelta(offset, "D")).day == end_day)
                    & [
                        ((
                            edges_base - pd.Timedelta(offset, "D")
                        ).month_name()[t][:3] == end_month)
                        for t in range(edges_base.size)
                    ]
                )
            ).dropna()
        else:
            raise Exception(
                f"group as str must be nD, pentad, 8day, dekad, 16day, nM, d-d Mmm"
                f" or d Mmm - d Mmm"
            )
    elif isinstance(group, int) :
        bins = group
    elif insintance(group, pandas.core.indexes.datetimes.DatetimeIndex):
        # custom bins edges from input
        bins = group
    else :
        raise Exception(
            f"group must be int, array, or str of form nD, pentad, 8day, dekad,"
            f" 16day, nM,d-d Mmm or d Mmm - d Mmm"
        )
    if (not isinstance(group, int)):
        assert (bins.size > 1), (
            "data must span at least one full group (need 2 edges to form 1 bin)"
        )
    return time_series.groupby_bins(time_series[time_dim], bins, right=False)


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
    seasonally_labeled_daily_data = daily_tobegroupedby_season(
        daily_rain, search_start_day, search_start_month, end_day, end_month
    )
    # Apply onset_date
    seasonal_data = (
        seasonally_labeled_daily_data[daily_rain.name]
        .groupby(seasonally_labeled_daily_data["seasons_starts"])
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
    seasons_ends = seasonally_labeled_daily_data["seasons_ends"].rename({"group": time_dim})
    seasonal_onset_date = xr.merge([seasonal_data, seasons_ends])

    # Tip to get dates from timedelta search_start_day
    # seasonal_onset_date = seasonal_onset_date[time_dim]
    # + seasonal_onset_date.onset_delta
    return seasonal_onset_date

def seasonal_cess_date_from_sm(
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
    seasonally_labeled_daily_data = daily_tobegroupedby_season(
        soil_moisture, search_start_day, search_start_month, end_day, end_month
    )
    # Apply cess_date
    seasonal_data = (
        seasonally_labeled_daily_data[soil_moisture.name]
        .groupby(seasonally_labeled_daily_data["seasons_starts"])
        .map(
            cess_date_from_sm,
            dry_thresh=dry_thresh,
            dry_spell_length_thresh=dry_spell_length_thresh,
        )
    ).rename("cess_delta")
    # Get the seasons ends
    seasons_ends = seasonally_labeled_daily_data["seasons_ends"].rename({"group": time_dim})
    seasonal_cess_date = xr.merge([seasonal_data, seasons_ends])

    # Tip to get dates from timedelta search_start_day
    # seasonal_onset_date = seasonal_onset_date[time_dim]
    # + seasonal_onset_date.onset_delta
    return seasonal_cess_date


def seasonal_cess_date_from_rain(
    daily_rain,
    search_start_day,
    search_start_month,
    search_days,
    dry_thresh,
    dry_spell_length_thresh,
    et,
    taw,
    sminit,
    time_dim="T"
):
    # Deal with leap year cases
    if search_start_day == 29 and search_start_month == 2:
        search_start_day = 1
        search_start_month = 3

    # Find an acceptable end_day/_month
    first_end_date = sel_day_and_month(
        daily_rain[time_dim], search_start_day, search_start_month
    )[0] + np.timedelta64(
        search_days,
        "D",
    )

    end_day = first_end_date.dt.day.values

    end_month = first_end_date.dt.month.values

    # Apply daily grouping by season
    seasonally_labeled_daily_data = daily_tobegroupedby_season(
        daily_rain, search_start_day, search_start_month, end_day, end_month
    )
    # Apply cess_date
    seasonal_data = (
        seasonally_labeled_daily_data[daily_rain.name]
        .groupby(seasonally_labeled_daily_data["seasons_starts"])
        .map(
            cess_date_from_rain,
            dry_thresh=dry_thresh,
            dry_spell_length_thresh=dry_spell_length_thresh,
            et=et,
            taw=taw,
            sminit=sminit,
        )
    ).rename("cess_delta")
    # Get the seasons ends
    seasons_ends = seasonally_labeled_daily_data["seasons_ends"].rename({"group": time_dim})
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
    seasonally_labeled_daily_data = daily_tobegroupedby_season(
        daily_data, start_day, start_month, end_day, end_month
    )
    seasonal_data = (
        seasonally_labeled_daily_data[daily_data.name]
        .groupby(seasonally_labeled_daily_data["seasons_starts"])
        .sum(dim=time_dim, skipna=True, min_count=min_count)
        #        .rename({"seasons_starts": time_dim})
    )
    seasons_ends = seasonally_labeled_daily_data["seasons_ends"].rename({"group": time_dim})
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
