import xarray as xr
import agronomy
import pandas as pd
import numpy as np


def test_spwbu_basic():

    sm_previous_day = xr.DataArray(30)
    peffective = xr.DataArray(10)
    et = xr.DataArray(5)
    taw = xr.DataArray(60)
    sm, drainage = agronomy.soil_plant_water_step(
        sm_previous_day,
        peffective,
        et,
        taw,
    )
    
    assert drainage == 0
    assert sm == 35
    
    
def test_spwb_with_dims_and_drainage():

    sm_previous_day = xr.DataArray([30, 56])
    peffective = xr.DataArray([10, 10])
    et = xr.DataArray([5, 5])
    taw = xr.DataArray([60, 60])
    sm, drainage = agronomy.soil_plant_water_step(
        sm_previous_day,
        peffective,
        et,
        taw,
    )

    assert (drainage == [0, 1]).all()
    assert (sm == [35, 60]).all()


def test_spwba_basic():
    
    t = pd.date_range(start="2000-05-01", end="2000-06-30", freq="1D")
    precip = xr.DataArray(np.ones(t.shape)*6, dims=["T"], coords={"T": t})
    sm, drainage, et_crop, et_crop_red, p_d = agronomy.soil_plant_water_balance(
        precip,
        et=5,
        taw=60,
        sminit=10,
    )
    assert np.allclose(sm[0:49], np.arange(49)+11)
    assert np.allclose(sm[49:], 60)
    assert np.allclose(drainage[0:49], 0)
    assert np.allclose(drainage[50:], 1)    
    assert np.allclose(et_crop, 5)
    assert np.allclose(et_crop_red, et_crop)
    assert p_d is None
    
    
def test_spwba_kc_pd():
    kc_periods = pd.TimedeltaIndex([0, 4, 5, 10, 10], unit="D")
    kc_params = xr.DataArray(
        data=[0.1, 0.5, 1., 1., 0.25], dims=["kc_periods"], coords=[kc_periods]
    )
    planting_date = xr.DataArray(
        pd.DatetimeIndex(data=["2000-05-02"]),
        dims=["Farm"], coords={"Farm": [0]}
    )
    t = pd.date_range(start="2000-05-01", end="2000-06-30", freq="1D")
    precip = xr.DataArray(np.ones(t.shape)*10, dims=["T"], coords={"T": t})
    sm, drainage, et_crop, et_crop_red, p_d = agronomy.soil_plant_water_balance(
        precip,
        et=10,
        taw=60,
        sminit=10,
        kc_params=kc_params,
        planting_date=planting_date,
    )

    assert np.allclose(
        et_crop.isel(T=slice(0,11)).squeeze(),
        [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    assert np.allclose(et_crop_red, et_crop)
    assert np.allclose(
        sm.isel(T=slice(0,11)).squeeze(),
        [10, 19, 27, 34, 40, 45, 49, 52, 54, 55, 55]
    )
    assert np.allclose(drainage.isel(T=slice(0,11)).squeeze(), 0)
    assert p_d == planting_date
    

def test_spwba_kc_pd_ks():
    kc_periods = pd.TimedeltaIndex([0, 4, 5, 10, 10], unit="D")
    kc_params = xr.DataArray(
        data=[0.1, 0.5, 1., 1., 0.25], dims=["kc_periods"], coords=[kc_periods]
    )
    planting_date = xr.DataArray(
        pd.DatetimeIndex(data=["2000-05-02"]),
        dims=["Farm"], coords={"Farm": [0]}
    )
    t = pd.date_range(start="2000-05-01", end="2000-06-30", freq="1D")
    precip = xr.DataArray(np.ones(t.shape)*10, dims=["T"], coords={"T": t})
    sm, drainage, et_crop, et_crop_red, p_d = agronomy.soil_plant_water_balance(
        precip,
        et=10,
        taw=60,
        sminit=10,
        kc_params=kc_params,
        planting_date=planting_date,
        rho_crop=1/3,
    )

    assert np.allclose(
        et_crop.isel(T=slice(0,11)).squeeze(),
        [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    assert np.allclose(
        et_crop_red.isel(T=slice(0,2)).squeeze(),
        [5, 0.75],
    )
    assert np.allclose(
        sm.isel(T=slice(0,2)).squeeze(),
        [15, 24.25],
    )


def test_spwba_kc_pd_ks_adj():
    kc_periods = pd.TimedeltaIndex([0, 4, 5, 10, 10], unit="D")
    kc_params = xr.DataArray(
        data=[0.1, 0.5, 1., 1., 0.25], dims=["kc_periods"], coords=[kc_periods]
    )
    planting_date = xr.DataArray(
        pd.DatetimeIndex(data=["2000-05-02"]),
        dims=["Farm"], coords={"Farm": [0]}
    )
    t = pd.date_range(start="2000-05-01", end="2000-06-30", freq="1D")
    precip = xr.DataArray(np.ones(t.shape)*10, dims=["T"], coords={"T": t})
    sm, drainage, et_crop, et_crop_red, p_d = agronomy.soil_plant_water_balance(
        precip,
        et=10,
        taw=60,
        sminit=10,
        kc_params=kc_params,
        planting_date=planting_date,
        rho_crop=0.7,
        rho_adj=True,
    )

    assert np.allclose(
        et_crop.isel(T=slice(0,11)).squeeze(),
        [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    assert np.allclose(et_crop_red.isel(T=0), 10/3)
    assert np.allclose(sm.isel(T=0), 20 - 10/3)


def test_spwba_findpd():
    kc_periods = pd.TimedeltaIndex([0, 45, 47, 45, 45], unit="D")
    kc_params = xr.DataArray(
        data=[0.2, 0.4, 1.2, 1.2, 0.6], dims=["kc_periods"], coords=[kc_periods]
    )
    sminit = xr.DataArray(
        data=[10, 20], dims=["X"], coords={"X": [0, 1]},
    )
    t = pd.date_range(start="2000-05-01", end="2000-06-30", freq="1D")
    precip = xr.DataArray(np.ones(t.shape)*6, dims=["T"], coords={"T": t})
    sm, drainage, et_crop, et_crop_red, p_d = agronomy.soil_plant_water_balance(
        precip,
        et=5,
        taw=60,
        sminit=sminit,
        kc_params=kc_params,
        sm_threshold=20,
    )

    assert (p_d[0] == precip["T"][10])
    assert (p_d[1] == precip["T"][0])


def test_antedecedent_precip_ind():
    t = pd.date_range(start="2000-05-01", end="2000-05-07", freq="1D")
    x = xr.DataArray(
        np.array([
            [6, 5, 4, 3, 2, 1, 2],
            [1, 1, 1, 1, 1, 1, 1],
        ]),
        dims=["X", "T"], coords={"X": [0, 1], "T": t}
    ) 
    api = agronomy.antecedent_precip_ind(x, 7).dropna("T").squeeze("T")

    assert np.allclose(api, [[7, 1/6 + 1/5 + 1/4 + 1/3 + 1/2 + 1 + 1/2 ]])


def test_api_runoff():
    t = pd.date_range(start="2000-05-01", end="2000-05-05", freq="1D")
    precip = xr.DataArray(np.arange(5), dims=["T"], coords={"T": t})
    api = agronomy.antecedent_precip_ind(precip, 2)
    runoff = agronomy.api_runoff(
        precip,
        api,
        no_runoff=1.5,
        api_thresh=(3, 4),
        api_poly=([1, 1, 1], [1, 2, 3], [-2, 0, 1])
    )

    assert np.allclose (api, [np.nan, 0.5, 2, 3.5, 5], equal_nan=True)
    assert np.allclose(runoff, [0, 1 + 1*2 + 1*2**2, 1 + 2*3 + 3*3**2, -2 + 0*4 + 1*4**2])


def test_solar_radiation():
    t = xr.DataArray(
        pd.date_range(start="2000-06-21", end="2000-12-21", freq="7D"),
        dims=["T"]
    )
    doy = t.dt.dayofyear
    lat = xr.DataArray(np.arange(40, 80, 5), dims=["Y"])
    lat = lat * np.pi / 180
    ra = agronomy.solar_radiation(doy, lat).dropna("Y")

    # RA decreases with length of days
    assert (ra.diff("T") <= 0).all()
    # Ra decreases with latitude in Winter
    assert (ra.where(lambda x: x["T"].dt.month >= 9, drop=True).diff("Y") <= 0).all()


def test_hargreaves_et_ref():
    et_ref = agronomy.hargreaves_et_ref(
        xr.DataArray(32),
        xr.DataArray(5),
        xr.DataArray(40)
    )

    assert et_ref == 0.0023 * (32 + 17.8) * np.sqrt(5) * 0.408 * 40
