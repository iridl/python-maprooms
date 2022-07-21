import numpy as np
import pandas as pd
import xarray as xr
import calc
import data_test_calc


def test_longest_run_length():

    precip = precip_sample()
    data_cond = precip < 1
    lds = calc.longest_run_length(data_cond, "T")

    assert lds == 9


def test_longest_run_length_where_last_is_flagged_and_makes_longest_spell():

    precip = precip_sample()
    precip[slice(-9, None)] = 0.1
    data_cond = precip < 1
    lds = calc.longest_run_length(data_cond, "T")

    assert lds == 14


def test_longest_run_length_all_flagged():

    precip = precip_sample() * 0 + 0.1
    data_cond = precip < 1
    lds = calc.longest_run_length(data_cond, "T")

    assert lds == precip["T"].size


def test_longest_run_length_all_unflagged():

    precip = precip_sample() * 0 + 2
    data_cond = precip < 1
    lds = calc.longest_run_length(data_cond, "T")

    assert lds == 0


def test_longest_run_length_2d():

    precip = xr.concat(
        [precip_sample(), precip_sample()[::-1].assign_coords(T=precip_sample()["T"])],
        dim="stn_id",
    )
    data_cond = precip < 1
    lds = calc.longest_run_length(data_cond, "T")

    assert (lds == [9, 9]).all()


def test_longest_run_length_where_first_is_flagged():

    data_cond = precip_sample()
    data_cond[:] = 1
    data_cond[55] = 0
    lds = calc.longest_run_length(data_cond, "T")

    assert lds == 55


def test_longest_run_length_coord_size_1():

    data_cond = (precip_sample()[0] > 0) * 1
    data_cond0 = data_cond - 1
    lds = calc.longest_run_length(data_cond, "T")
    lds0 = calc.longest_run_length(data_cond0, "T")

    assert lds == 1
    assert lds0 == 0


def test_following_dry_spell_length():

    precip = precip_sample()
    dsl = calc.following_dry_spell_length(precip, 1)
    expected = [5., 4., 3., 2., 1., 0., 0., 0., 0., 0., 0., 0., 4., 3., 2., 1., 0.,
       2., 1., 0., 1., 0., 2., 1., 0., 0., 9., 8., 7., 6., 5., 4., 3., 2.,
       1., 0., 0., 0., 0., 2., 1., 0., 0., 2., 1., 0., 5., 4., 3., 2., 1.,
       0., 0., 0., 3., 2., 1., 0., 0., 0.]
    precip[0] = 2
    dsl_wet_head = calc.following_dry_spell_length(precip, 1)
    precip[-1] = 0.1
    dsl_dry_tail = calc.following_dry_spell_length(precip, 1)

    assert np.array_equal(dsl, expected)
    assert np.array_equal(dsl_wet_head, expected)
    expected[-1] = 1
    assert np.array_equal(dsl_dry_tail, expected)


def test_following_dry_spell_length_heading_wet_and_tailing_dry():

    precip = precip_sample()[::-1].assign_coords(T=precip_sample()["T"])
    dsl = calc.following_dry_spell_length(precip, 1)
    expected = [0., 0., 3., 2., 1., 0., 0., 0., 5., 4., 3., 2., 1., 0., 2., 1., 0.,
       0., 2., 1., 0., 0., 0., 0., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
       0., 2., 1., 0., 1., 0., 2., 1., 0., 4., 3., 2., 1., 0., 0., 0., 0.,
       0., 0., 0., 6., 5., 4., 3., 2., 1.]

    assert np.array_equal(dsl, expected)


def test_sel_day_and_month_1yr():

    precip = precip_sample()
    dam = calc.sel_day_and_month(precip["T"], 6, 5)

    assert dam.values[0] == pd.to_datetime("2000-05-06T00:00:00.000000000")
    

def test_sel_day_and_month_6yrs():

    precip = data_test_calc.multi_year_data_sample()
    dam = calc.sel_day_and_month(precip["T"], 19, 1)
    
    assert (dam.values == pd.to_datetime([
        "2000-01-19T00:00:00.000000000",
        "2001-01-19T00:00:00.000000000",
        "2002-01-19T00:00:00.000000000",
        "2003-01-19T00:00:00.000000000",
        "2004-01-19T00:00:00.000000000",
        "2005-01-19T00:00:00.000000000",
    ])).all()


def test_sel_day_and_month_with_offset():

    precip = data_test_calc.multi_year_data_sample()
    dam = calc.sel_day_and_month(precip["T"], 1, 3, offset=-1)
    
    assert (dam.values == pd.to_datetime([
        "2000-02-29T00:00:00.000000000",
        "2001-02-28T00:00:00.000000000",
        "2002-02-28T00:00:00.000000000",
        "2003-02-28T00:00:00.000000000",
        "2004-02-29T00:00:00.000000000",
        "2005-02-28T00:00:00.000000000",
    ])).all()


def test_water_balance_intializes_right():

    precip = precip_sample()
    wb = calc.water_balance(precip, 5, 60, 0)

    assert wb.soil_moisture.isel(T=0) == 0


def test_water_balance():

    precip = precip_sample()
    wb = calc.water_balance(precip, 5, 60, 0)

    assert np.allclose(wb.soil_moisture.isel(T=-1), 10.350632)


def test_water_balance2():

    t = pd.date_range(start="2000-05-01", end="2000-05-04", freq="1D")
    values = [
        [5.0, 6.0, 3.0, 66.0],
        [10.0, 12.0, 14.0, 16.0],
    ]
    precip = xr.DataArray(values, dims=["X", "T"], coords={"T": t})
    wb = calc.water_balance(precip, 5, 60, 0)

    assert np.array_equal(wb.soil_moisture["T"], t)
    expected = [
        [0.0, 1.0, 0.0, 60.0],
        [5.0, 12.0, 21.0, 32.0],
    ]
    assert np.array_equal(wb.soil_moisture, expected)


def test_water_balance_et_is_xarray_but_has_no_T():

    t = pd.date_range(start="2000-05-01", end="2000-05-04", freq="1D")
    values = [
        [5.0, 6.0, 3.0, 66.0],
        [10.0, 12.0, 14.0, 16.0],
    ]
    precip = xr.DataArray(values, dims=["X", "T"], coords={"T": t})
    et = xr.DataArray([5, 10], dims=["X"])
    wb = calc.water_balance(precip, et, 60, 0)

    assert np.array_equal(wb.soil_moisture["T"], t)
    expected = [
        [0.0, 1.0, 0.0, 60.0],
        [0.0, 2.0, 6.0, 12.0],
    ]
    assert np.array_equal(wb.soil_moisture, expected)


def test_water_balance_et_has_T():

    t = pd.date_range(start="2000-05-01", end="2000-05-04", freq="1D")
    values = [
        [5.0, 6.0, 3.0, 66.0],
        [10.0, 12.0, 14.0, 16.0],
    ]
    precip = xr.DataArray(values, dims=["X", "T"], coords={"T": t})
    values = [5.0, 10.0, 15.0, 10.0]
    et = xr.DataArray(values, dims=["T"], coords={"T": t})
    wb = calc.water_balance(precip, et, 60, 0)

    assert np.array_equal(wb.soil_moisture["T"], t)
    expected = [
        [0.0, 0.0, 0.0, 56.0],
        [5.0, 7.0, 6.0, 12.0],
    ]
    assert np.array_equal(wb.soil_moisture, expected)


def test_daily_tobegroupedby_season_cuts_on_days():

    precip = data_test_calc.multi_year_data_sample()
    dts = calc.daily_tobegroupedby_season(precip, 29, 11, 29, 2)

    assert dts["T"].size == 461


def test_daily_tobegroupedby_season_creates_groups():

    precip = data_test_calc.multi_year_data_sample()
    dts = calc.daily_tobegroupedby_season(precip, 29, 11, 29, 2)

    assert dts["group"].size == 5


def test_daily_tobegroupedby_season_picks_right_end_dates():

    precip = data_test_calc.multi_year_data_sample()
    dts = calc.daily_tobegroupedby_season(precip, 29, 11, 29, 2)
    assert (
        dts.seasons_ends
        == pd.to_datetime(
            [
                "2001-02-28T00:00:00.000000000",
                "2002-02-28T00:00:00.000000000",
                "2003-02-28T00:00:00.000000000",
                "2004-02-29T00:00:00.000000000",
                "2005-02-28T00:00:00.000000000",
            ],
        )
    ).all()


def test_seasonal_onset_date_keeps_returning_same_outputs():

    precip = data_test_calc.multi_year_data_sample()
    onsetsds = calc.seasonal_onset_date(
        daily_rain=precip,
        search_start_day=1,
        search_start_month=3,
        search_days=90,
        wet_thresh=1,
        wet_spell_length=3,
        wet_spell_thresh=20,
        min_wet_days=1,
        dry_spell_length=7,
        dry_spell_search=21,
        time_coord="T",
    )
    onsets = onsetsds.onset_delta + onsetsds["T"]
    assert np.array_equal(
        onsets,
        pd.to_datetime(
            [
                "NaT",
                "2001-03-08T00:00:00.000000000",
                "NaT",
                "2003-04-12T00:00:00.000000000",
                "2004-04-04T00:00:00.000000000",
            ],
        ),
        equal_nan=True,
    )

def test_seasonal_cess_date_keeps_returning_same_outputs():

    precip = data_test_calc.multi_year_data_sample()
    wb = calc.water_balance(
        daily_rain=precip,
        et=5,
        taw=60,
        sminit=0,
        time_coord="T"
    ).to_array(name="soil moisture")
    cessds = calc.seasonal_cess_date(
        soil_moisture=wb,
        search_start_day=1,
        search_start_month=9,
        search_days=90,
        dry_thresh=5,
        min_dry_days=3,
        time_coord="T"
    )
    cess = (cessds.cess_delta + cessds["T"]).squeeze()
    assert np.array_equal(
        cess,
        pd.to_datetime(
            [
                "2000-09-21T00:00:00.000000000",
                "2001-09-03T00:00:00.000000000",
                "2002-09-03T00:00:00.000000000",
                "2003-09-24T00:00:00.000000000",
                "2004-09-01T00:00:00.000000000",
            ],
        ),
        equal_nan=True,  
    )

def test_seasonal_onset_date():
    t = pd.date_range(start="2000-01-01", end="2005-02-28", freq="1D")
    # this is rr_mrg.sel(T=slice("2000", "2005-02-28")).isel(X=150, Y=150).precip
    synthetic_precip = xr.DataArray(np.zeros(t.size), dims=["T"], coords={"T": t}) + 1.1
    synthetic_precip = xr.where(
        (synthetic_precip["T"] == pd.to_datetime("2000-03-29"))
        | (synthetic_precip["T"] == pd.to_datetime("2000-03-30"))
        | (synthetic_precip["T"] == pd.to_datetime("2000-03-31"))
        | (synthetic_precip["T"] == pd.to_datetime("2001-04-30"))
        | (synthetic_precip["T"] == pd.to_datetime("2001-05-01"))
        | (synthetic_precip["T"] == pd.to_datetime("2001-05-02"))
        | (synthetic_precip["T"] == pd.to_datetime("2002-04-01"))
        | (synthetic_precip["T"] == pd.to_datetime("2002-04-02"))
        | (synthetic_precip["T"] == pd.to_datetime("2002-04-03"))
        | (synthetic_precip["T"] == pd.to_datetime("2003-05-16"))
        | (synthetic_precip["T"] == pd.to_datetime("2003-05-17"))
        | (synthetic_precip["T"] == pd.to_datetime("2003-05-18"))
        | (synthetic_precip["T"] == pd.to_datetime("2004-03-01"))
        | (synthetic_precip["T"] == pd.to_datetime("2004-03-02"))
        | (synthetic_precip["T"] == pd.to_datetime("2004-03-03")),
        7,
        synthetic_precip,
    ).rename("synthetic_precip")

    onsetsds = calc.seasonal_onset_date(
        daily_rain=synthetic_precip,
        search_start_day=1,
        search_start_month=3,
        search_days=90,
        wet_thresh=1,
        wet_spell_length=3,
        wet_spell_thresh=20,
        min_wet_days=1,
        dry_spell_length=7,
        dry_spell_search=21,
        time_coord="T",
    )
    onsets = onsetsds.onset_delta + onsetsds["T"]
    assert (
        onsets
        == pd.to_datetime(
            xr.DataArray(
                [
                    "2000-03-29T00:00:00.000000000",
                    "2001-04-30T00:00:00.000000000",
                    "2002-04-01T00:00:00.000000000",
                    "2003-05-16T00:00:00.000000000",
                    "2004-03-01T00:00:00.000000000",
                ],
                dims=["T"],
                coords={"T": onsets["T"]},
            )
        )
    ).all()

def test_seasonal_cess_date():
    t = pd.date_range(start="2000-01-01", end="2005-02-28", freq="1D")
    synthetic_precip = xr.DataArray(np.zeros(t.size), dims=["T"], coords={"T": t}) + 1.1
    synthetic_precip = xr.where(
        (synthetic_precip["T"] == pd.to_datetime("2000-03-29"))
        | (synthetic_precip["T"] == pd.to_datetime("2000-03-30"))
        | (synthetic_precip["T"] == pd.to_datetime("2000-03-31"))
        | (synthetic_precip["T"] == pd.to_datetime("2001-04-30"))
        | (synthetic_precip["T"] == pd.to_datetime("2001-05-01"))
        | (synthetic_precip["T"] == pd.to_datetime("2001-05-02"))
        | (synthetic_precip["T"] == pd.to_datetime("2002-04-01"))
        | (synthetic_precip["T"] == pd.to_datetime("2002-04-02"))
        | (synthetic_precip["T"] == pd.to_datetime("2002-04-03"))
        | (synthetic_precip["T"] == pd.to_datetime("2003-05-16"))
        | (synthetic_precip["T"] == pd.to_datetime("2003-05-17"))
        | (synthetic_precip["T"] == pd.to_datetime("2003-05-18"))
        | (synthetic_precip["T"] == pd.to_datetime("2004-03-01"))
        | (synthetic_precip["T"] == pd.to_datetime("2004-03-02"))
        | (synthetic_precip["T"] == pd.to_datetime("2004-03-03")),
        7,
        synthetic_precip,
    ).rename("synthetic_precip")    

    wb = calc.water_balance(
        daily_rain=synthetic_precip,
        et=5,
        taw=60,
        sminit=0,
        time_coord="T"
    ).to_array(name="soil moisture")
    cessds = calc.seasonal_cess_date(
        soil_moisture=wb,
        search_start_day=1,
        search_start_month=9,
        search_days=90,
        dry_thresh=5,
        min_dry_days=3,
        time_coord="T"
    )
    cess = (cessds.cess_delta + cessds["T"]).squeeze()
    assert (
        cess
        == pd.to_datetime(
            xr.DataArray(
                [
                    "2000-09-01T00:00:00.000000000",
                    "2001-09-01T00:00:00.000000000",
                    "2002-09-01T00:00:00.000000000",
                    "2003-09-01T00:00:00.000000000",
                    "2004-09-01T00:00:00.000000000",
                ],dims=["T"],coords={"T": cess["T"]},
            )
        )
    ).all()


def precip_sample():

    t = pd.date_range(start="2000-05-01", end="2000-06-30", freq="1D")
    # this is rr_mrg.isel(X=0, Y=124, drop=True).sel(T=slice("2000-05-01", "2000-06-30"))
    # fmt: off
    values = [
        0.054383,  0.      ,  0.      ,  0.027983,  0.      ,  0.      ,
        7.763758,  3.27952 , 13.375934,  4.271866, 12.16503 ,  9.706059,
        7.048605,  0.      ,  0.      ,  0.      ,  0.872769,  3.166048,
        0.117103,  0.      ,  4.584551,  0.787962,  6.474878,  0.      ,
        0.      ,  2.834413,  9.029134,  0.      ,  0.269645,  0.793965,
        0.      ,  0.      ,  0.      ,  0.191243,  0.      ,  0.      ,
        4.617332,  1.748801,  2.079067,  2.046696,  0.415886,  0.264236,
        2.72206 ,  1.153666,  0.204292,  0.      ,  5.239006,  0.      ,
        0.      ,  0.      ,  0.      ,  0.679325,  2.525344,  2.432472,
        10.737132,  0.598827,  0.87709 ,  0.162611, 18.794922,  3.82739 ,
        2.72832
    ]
    # fmt: on
    precip = xr.DataArray(values, dims=["T"], coords={"T": t})
    return precip


def call_onset_date(data):
    onsets = calc.onset_date(
        daily_rain=data,
        wet_thresh=1,
        wet_spell_length=3,
        wet_spell_thresh=20,
        min_wet_days=1,
        dry_spell_length=7,
        dry_spell_search=21,
    )
    return onsets


def call_cess_date(data):
    cessations = calc.cess_date(
        soil_moisture=data,
        dry_thresh=5,
        min_dry_days=3,
    )
    return cessations


def test_onset_date():

    precip = precip_sample()
    onsets = call_onset_date(precip)
    
    assert pd.Timedelta(onsets.values) == pd.Timedelta(days=6)
    # Converting to pd.Timedelta doesn't change the meaning of the
    # assertion, but gives a more helpful error message when the test
    # fails: Timedelta('6 days 00:00:00')
    # vs. numpy.timedelta64(518400000000000,'ns')


def test_onset_date_no_dry_spell():

    precip = precip_sample()
    onsets = calc.onset_date(
        daily_rain=precip,
        wet_thresh=1,
        wet_spell_length=3,
        wet_spell_thresh=20,
        min_wet_days=1,
        dry_spell_length=4,
        dry_spell_search=0,
    )
    
    assert pd.Timedelta(onsets.values) == pd.Timedelta(days=6)

def test_cess_date():

    sm = precip_sample() + 4.95
    cessations = call_cess_date(sm)
    assert pd.Timedelta(cessations.values) == pd.Timedelta(days=1)


def test_onset_date_with_other_dims():

    precip = xr.concat(
        [precip_sample(), precip_sample()[::-1].assign_coords(T=precip_sample()["T"])],
        dim="dummy_dim",
    )
    onsets = call_onset_date(precip)
    assert (
        onsets
        == xr.DataArray(
            [pd.Timedelta(days=6), pd.Timedelta(days=0)],
            dims=["dummy_dim"],
            coords={"dummy_dim": onsets["dummy_dim"]},
        )
    ).all()


def test_cess_date_with_other_dims():

    sm = xr.concat(
        [precip_sample(), precip_sample()[::-1].assign_coords(T=precip_sample()["T"])],
        dim="dummy_dim",
    )
    cessations = call_cess_date(sm)
    assert (
        cessations
        == xr.DataArray(
            [pd.Timedelta(days=0), pd.Timedelta(days=3)],
            dims=["dummy_dim"],
            coords={"dummy_dim": cessations["dummy_dim"]},
        )
    ).all()


def test_onset_date_returns_nat():

    precip = precip_sample()
    precipNaN = precip + np.nan
    onsetsNaN = call_onset_date(precipNaN)
    assert np.isnat(onsetsNaN.values)


def test_cess_date_returns_nat():

    sm = precip_sample()
    smNaN = sm + np.nan
    cessNaN = call_cess_date(smNaN)
    assert np.isnat(cessNaN.values)


def test_onset_date_dry_spell_invalidates():

    precip = precip_sample()
    precipDS = xr.where(
        (precip["T"] > pd.to_datetime("2000-05-09"))
        & (precip["T"] < (pd.to_datetime("2000-05-09") + pd.Timedelta(days=5))),
        0,
        precip,
    )
    onsetsDS = call_onset_date(precipDS)
    assert pd.Timedelta(onsetsDS.values) != pd.Timedelta(days=6)

def test_cess_date_wet_spell_invalidates():
  
    precip = precip_sample()
    precipDS = xr.where((precip["T"] > pd.to_datetime("2000-05-02")), 5, precip)
    cessDS = call_cess_date(precipDS)
    assert pd.Timedelta(cessDS.values) != pd.Timedelta(days=0)

def test_onset_date_late_dry_spell_invalidates_not():

    precip = precip_sample()
    preciplateDS = xr.where(
        (precip["T"] > (pd.to_datetime("2000-05-09") + pd.Timedelta(days=20))),
        0,
        precip,
    )
    onsetslateDS = call_onset_date(preciplateDS)
    assert pd.Timedelta(onsetslateDS.values) == pd.Timedelta(days=6)


def test_onset_date_1st_wet_spell_day_not_wet_day():
    """May 4th is 0.28 mm thus not a wet day
    resetting May 5th and 6th respectively to 1.1 and 18.7 mm
    thus, May 5-6 are both wet days and need May 4 to reach 20mm
    but the 1st wet day of the spell is not 4th but 5th
    """

    precip = precip_sample()
    precipnoWD = xr.where(
        (precip["T"] == pd.to_datetime("2000-05-05")),
        1.1,
        precip,
    )
    precipnoWD = xr.where(
        (precip["T"] == pd.to_datetime("2000-05-06")),
        18.7,
        precipnoWD,
    )
    onsetsnoWD = call_onset_date(precipnoWD)
    assert pd.Timedelta(onsetsnoWD.values) == pd.Timedelta(days=4)


def test_probExceed():
    earlyStart = pd.to_datetime(f"2000-06-01", yearfirst=True)
    values = {
        "onset": [
            "2000-06-18",
            "2000-06-16",
            "2000-06-26",
            "2000-06-01",
            "2000-06-15",
            "2000-06-07",
            "2000-07-03",
            "2000-06-01",
            "2000-06-26",
            "2000-06-01",
            "2000-06-08",
            "2000-06-23",
            "2000-06-01",
            "2000-06-01",
            "2000-06-16",
            "2000-06-02",
            "2000-06-17",
            "2000-06-18",
            "2000-06-10",
            "2000-06-17",
            "2000-06-05",
            "2000-06-07",
            "2000-06-03",
            "2000-06-10",
            "2000-06-17",
            "2000-06-05",
            "2000-06-11",
            "2000-06-01",
            "2000-06-24",
            "2000-06-06",
            "2000-06-07",
            "2000-06-17",
            "2000-06-14",
            "2000-06-20",
            "2000-06-17",
            "2000-06-14",
            "2000-06-23",
            "2000-06-01",
        ]
    }
    onsetMD = pd.DataFrame(values).astype("datetime64[ns]")
    cumsum = calc.probExceed(onsetMD, earlyStart)
    probExceed_values = [
        0.815789,
        0.789474,
        0.763158,
        0.710526,
        0.684211,
        0.605263,
        0.578947,
        0.526316,
        0.500000,
        0.447368,
        0.421053,
        0.368421,
        0.236842,
        0.184211,
        0.157895,
        0.105263,
        0.078947,
        0.026316,
        0.000000,
    ]
    assert np.allclose(cumsum.probExceed, probExceed_values)
