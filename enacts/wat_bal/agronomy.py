import xarray as xr
import numpy as np


DEFAULT_API_THRESHOLD = (6.3, 19, 31.7, 44.4, 57.1, 69.8)
DEFAULT_API_POLYNOMIALS = (
    (0.858, 0.0895, 0.0028),
    (-1.14, 0.042, 0.0026),
    (-2.34, 0.12, 0.0026),
    (-2.36, 0.19, 0.0026),
    (-2.78, 0.25, 0.0026),
    (-3.17, 0.32, 0.0024),
    (-4.21, 0.438, 0.0018),
)


def soil_plant_water_step(
    sm_yesterday,
    peffective,
    et,
    taw,
):
    """Compute soil-plant-water balance from yesterday to today.
    The balance is thought as a bucket with water coming in and out:
    
    `sm` (t) + `drainage` (t) = `sm` (t-1) + `peffective` (t) - `et` (t)
    
    where:
    
    `sm` is the soil moisture and can not exceed total available water `taw`.
    
    `drainage` is the residual soil moisture occasionally exceeding `taw`
    that drains through the soil.
    
    `peffective` is the effective precipitation that enters the soil.
    
    `et` is the evapotranspiration yielded by the plant.
    
    Parameters
    ----------
    sm_yesterday : DataArray
        soil moisture of yesterday.
    peffective : DataArray
        effective precipitation today.
    et : DataArray
        evapotranspiration of the plant today.
    taw : DataArray
        total available water that represents the maximum water capacity of the soil.
        
    Returns
    -------
    sm, drainage : Tuple of DataArray
        today soil moisture and drainage
        
    See Also
    --------
    soil_plant_water_balance
    
    """
    
    # Water Balance
    wb = (sm_yesterday + peffective - et).clip(min=0)
    drainage = (wb - taw).clip(min=0)
    sm = wb - drainage
    return sm, drainage


def soil_plant_water_balance(
    peffective,
    et,
    taw,
    sminit,
    kc_params=None,
    planting_date=None,
    sm_threshold=None,
    time_dim="T",
):
    """Compute soil-plant-water balance day after day over a growing season.
    See `soil_plant_water_step` for the step by step algorithm definition.
    The daily evapotranspiration `et` can be scaled by a Crop Cultivar Kc
    modelizing a crop needs in water according to the stage of its growth.
    Kc is set to 1 outside of the growing period. i.e. before planting date
    and after the last Kc curve inflection point. The planting date can either be
    prescribed as a parameter or evaluated by the simulation as the day following
    the day that soil moisture reached a cetain value for the first time.
    
    Parameters
    ----------
    peffective : DataArray
        Daily effective precipitation.
    et : DataArray
        Daily evapotranspiration of the plant.
    taw : DataArray
        Total available water that represents the maximum water capacity of the soil.
    sminit : DataArray
        Timeless soil moisture to initialize the loop with.
    kc_params : DataArray
        Crop Cultivar Kc parameters as a function of the inflection points of the Kc curve,
        expressed in consecutive daily time deltas originating from `planting_date`
        as coordinate `kc_periods` (default `kc_params` =None in which case Kc is set to 1).
    planting_date : DataArray[datetime64[ns]]
        Dates when planting (default `planting_date` =None in which case
        `planting_date` is assigned by the simulation according to a soil moisture
        criterion parametrizable through `sm_threshold` )
    sm_threshold : DataArray
        Planting the day after soil moisture is greater or equal to `sm_threshold`
        in units of soil moisture (default `sm_threshold` =None in which case
        `planting_date` must be defined)
    time_dim : str, optional
        Daily time dimension to run the balance against (default `time_dim` ="T").
        
    Returns
    -------
    sm, drainage, et_crop, planting_date : Tuple of DataArray
        Daily soil moisture, drainage and crop evapotranspiration over the growing season.
        Planting dates as given in parameters or as evaluated by the simulation.
        
    See Also
    --------
    soil_plant_water_step
    
    Examples
    --------
    Example of kc_params:
    
    >>> kc_periods = pd.TimedeltaIndex([0, 45, 47, 45, 45], unit="D")
    >>> kc_params = xr.DataArray(
    >>>    data=[0.2, 0.4, 1.2, 1.2, 0.6], dims=["kc_periods"], coords=[kc_periods]
    >>> )
    <xarray.DataArray (kc_periods: 5)>
    array([0.2, 0.4, 1.2, 1.2, 0.6])
    Coordinates:
        * kc_periods  (kc_periods) timedelta64[ns] 0 days 45 days ... 45 days 45 days
    
    Example of planting_date:
    
    >>> p_d = xr.DataArray(
    >>>    pd.DatetimeIndex(data=["2000-05-02", "2000-05-13"]),
    >>>    dims=["station"],
    >>>    coords={"station": [0, 1]},
    >>> )
    <xarray.DataArray (station: 2)>
    array(['2000-05-02T00:00:00.000000000', '2000-05-13T00:00:00.000000000'],
        dtype='datetime64[ns]')
    Coordinates:
        * station        (station) int64 0 1
    """
    
    # Initializations
    et = xr.DataArray(et)
    taw = xr.DataArray(taw)
    sminit = xr.DataArray(sminit)
    if kc_params is None:
        if planting_date is not None or sm_threshold is not None:
            raise Exception(
                "if Kc is not defined, neither planting_date nor sm_threshold should be"
            )
        kc = 1
        et_crop = et
    else: #et_crop depends on et, kc and planting_date dims, and time_dim
        kc_inflex = kc_params.assign_coords(
            kc_periods=kc_params["kc_periods"].cumsum(dim="kc_periods")
        )
        if planting_date is not None: # distance between 1st and planting days
            if sm_threshold is not None:
                raise Exception("either planting_date or sm_threshold should be defined")
            planted_since = peffective[time_dim][0].drop_vars(time_dim) - planting_date
        else: # 1st day is planting day if sminit met condition
            if sm_threshold is not None:
                planted_since = xr.where(
                    sminit >= sm_threshold, 0, np.nan
                ).astype("timedelta64[D]")
            else:
                raise Exception("if planting_date is not defined, then define a sm_threshold")
        et_crop = et.broadcast_like(
            peffective[time_dim]
        ).broadcast_like(
            planted_since
        ).broadcast_like(
            kc_params.isel({"kc_periods": 0}, drop=True)
        ) * np.nan
    # sminit depends on peffective, et_crop and taw dims, and the day before time_dim[0]
    sminit = sminit.broadcast_like(
        peffective.isel({time_dim: 0})
    ).broadcast_like(
        et_crop.isel({time_dim: 0}, missing_dims='ignore', drop=True)
    ).broadcast_like(
        taw
    ).assign_coords({time_dim: peffective[time_dim][0] - np.timedelta64(1, "D")})
    # sm depends on sminit dims and time_dim
    sm = sminit.drop_vars(time_dim).broadcast_like(peffective[time_dim]) * np.nan
    # drainage depends on sm dims
    drainage = xr.full_like(sm, fill_value=np.nan)
    # sm starts with initial condition sminit
    sm = xr.concat([sminit, sm], time_dim)
    # Filling/emptying bucket day after day
    for doy in range(0, peffective[time_dim].size):
        if kc_params is not None: # interpolate kc value per distance from planting
            kc = kc_inflex.interp(
                kc_periods=planted_since, kwargs={"fill_value": 1}
            ).where(lambda x: x.notnull(), other=1).drop_vars("kc_periods")
            if time_dim in et_crop.dims: # et _crop depends on time_dim but et might not
                et_crop[{time_dim: doy}] = kc * et.isel({time_dim: doy}, missing_dims='ignore')
        # water balance step
        sm[{time_dim: doy+1}], drainage[{time_dim: doy}] = soil_plant_water_step(
            sm.isel({time_dim: doy}, drop=True),
            peffective.isel({time_dim: doy}, drop=True),
            et_crop.isel({time_dim: doy}, missing_dims='ignore', drop=True),
            taw,
        )
        # Increment planted_since
        if kc_params is not None:
            if planting_date is None: # did doy met planting conditions?
                planted_since = planted_since.where(
                    lambda x: x.notnull(), # no planting date found yet
                    other=xr.where( # next day is planting if sm condition met
                        sm.isel({time_dim: doy+1}) >= sm_threshold, -1, np.nan
                    ).astype("timedelta64[D]"),
                )
            planted_since = planted_since + np.timedelta64(1, "D")
    # Let's have sm same shape as other variables
    sm = sm.isel({time_dim: slice(1,None)})
    # Let's save planting_date
    if kc_params is not None and planting_date is None:
        planting_date = (peffective[time_dim][-1].drop_vars(time_dim)
            - (planted_since - np.timedelta64(1, "D"))
        )
    return sm, drainage, et_crop, planting_date


def api_runoff(
    daily_rain,
    api,
    no_runoff=12.5,
    api_thresh=DEFAULT_API_THRESHOLD,
    api_poly=DEFAULT_API_POLYNOMIALS,
):
    """Computes Runoff based on an Antecedent Precipitation Index.
    `runoff` is a polynomial `api_poly` of `daily_rain`.
    Polynomial is chosen based on API categories defined by `api_thresh`.
    Additionaly, `runoff` is 0 if it rains less or equal than `no_runoff` ,
    and negative `runoff` is 0.

    Parameters
    ----------
    daily_rain : DataArray
        daily precipitation
    api : DataArray
        an Antecedent Precipitiona Index. Must be same size as daily_rain.
    no_runoff : DataArray, optional
        `runoff` is 0 if `daily_rain` is lesser or equal to `no_runoff`
        (default `no_runoff` =12.5)
    api_thresh : iterable, optional
        increasing API values
        indicating the upper limit (inclusive) to belong to an API category.
    api_poly : iterable(iterables), optional
        iterable of size one more than `api_thresh` 's of iterables of
        polynomial coefficients in order of increasing degree.
        The polynomial used to compute the `runoff` is picked according to the categories
        defined by the thresholds.
        
    Returns
    -------
    runoff : DataArray
        daily Runoff.

    See Also
    --------
    antecedent_precip_ind, numpy.polynomial.polynomial.Polynomial
    
    Notes
    -----
    `runoff` drops NA values.
    Typically the heading values of `api`
    that is typically defined on a rolling time window of daily rain.
    The default `api_thresh` is

    >>> (6.3, 19, 31.7, 44.4, 57.1, 69.8)

    and the default `api_poly` is

    >>> (
    >>>         (0.858, 0.0895, 0.0028),
    >>>         (-1.14, 0.042, 0.0026),
    >>>         (-2.34, 0.12, 0.0026),
    >>>         (-2.36, 0.19, 0.0026),
    >>>         (-2.78, 0.25, 0.0026),
    >>>         (-3.17, 0.32, 0.0024),
    >>>         (-4.21, 0.438, 0.0018),
    >>> )

    Which means for instance that, if rain is greater or equal to 12.5
    and API is 18, then runoff is
    
    -1.14 + 0.042*x 0.0026*(x**2)
    
    where x is daily rain.
    """
    conds = (
        [(daily_rain <= no_runoff).values] +
        [
            (
                (daily_rain > no_runoff) &
                ((i == 0) | (api > api_thresh[i-1])) &
                ((i == len(api_thresh)) | (api <= api_thresh[i]))
            ).values
            for i in range(len(api_thresh))
        ]
    )
    funcs = [0] + [
        np.polynomial.polynomial.Polynomial(coeffs)
        for coeffs in api_poly
    ]
    return xr.DataArray(
        np.piecewise(daily_rain, conds, funcs),
        dims=daily_rain.dims, attrs=dict(description="Runoff", units="mm")
    ).where(~np.isnan(api), drop=True).clip(min=0).rename("runoff")


def antecedent_precip_ind(daily_rain, n, time_dim="T"):
    """Antecedent Precipitation Index (API) is a rolling weighted sum
    of daily rainfall `daily_rain` over a window of `n` days.
    The weights are 1/2 for last day and 1/( `n` -i-1) for i :sup:`th` day of the window.

    Parameters
    ----------
    daily_rain : DataArray
        daily precipitation.
    n: int
        size of the rolling window to weight-sum against.
    time_dim : str, optional
        Daily time dimension to run weighted-sum against (default `time_dim` ="T").
    
    Returns
    -------
    api : DataArray
        weighted-sum of `daily_rain` along `time_dim` .

    See Also
    --------
    api_runoff
    """
    dr_rolled = daily_rain.rolling(**{time_dim: n}).construct("window")
    return dr_rolled.weighted(
        1 / dr_rolled["window"][::-1].where(lambda x: x != 0, 2)
    ).sum(dim="window", skipna=False).rename("api")


def hargreaves_et_ref(temp_avg, temp_amp, ra):
    """Computes Reference Evapotranspiration as a function of
    temperature (average and amplitude in Celsius) and solar radation
    by the Hargreaves method.

    Parameters
    ----------
    temp_avg : DataArray
        daily average temperature in Celsius
    temp_amp: DataArray
        daily temperature amplitude in Celsius
    ra : DataArray
        solar radiation in  MJ m−2 day−1

    Returns
    -------
    et_ref : DataArray
        daily reference evapotranspiration in mm.
    """
    # the Hargreaves coefficient.
    ah = 0.0023
    # the value of 0.408 is
    # the inverse of the latent heat flux of vaporization at 20C,
    # changing the extraterrestrial radiation units from MJ m−2 day−1
    # into mm day−1 of evaporation equivalent
    bh = 0.408
    et_ref = (
        ah * (temp_avg + 17.8) * np.sqrt(temp_amp) * bh * ra
    ).rename("et_ref")
    et_ref.attrs = dict(description="Reference Evapotranspiration", units="mm")
    return et_ref


def solar_radiation(doy, lat):
    """Computes solar radiation for day of year and latitude in radians
    
    Parameters
    ----------
    doy : DataArray
        day of the year from dt.dayofyear.
    lat : DataArray
        latitude in radians.

    Returns
    -------
    ra : DataArray
        solar radiation in MJ/m**2/day at latitude `lat` for day of the year `doy` .
    """
    distance_relative = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    solar_declination = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    sunset_hour_angle = np.arccos(-1 * np.tan(lat) * np.tan(solar_declination))
    ra = (
        24 * 60 * 0.082 * distance_relative
        * (
            sunset_hour_angle * np.sin(lat) * np.sin(solar_declination)
            + np.sin(sunset_hour_angle) * np.cos(lat) * np.cos(solar_declination)
        )
        / np.pi
    ).rename("ra")
    ra.attrs = dict(description="Extraterrestrial Radiation", units="MJ/m**2/day")
    return ra