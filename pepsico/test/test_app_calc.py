import numpy as np
import pandas as pd
import xarray as xr
import app_calc


def test__cumsum_flagged_diff():

    t = pd.date_range(start="2000-05-01", end="2000-05-10", freq="1D")
    values = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0])
    precip = xr.DataArray(values, coords={"T": t})
    spells = app_calc._cumsum_flagged_diff(precip, "T")

    np.testing.assert_array_equal(spells, [1, 2, 0, 0, 0, 3, 0, 0, 0])


def test__cumsum_flagged_diff_where_last_is_flagged():

    t = pd.date_range(start="2000-05-01", end="2000-05-10", freq="1D")
    values = np.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 1])
    precip = xr.DataArray(values, coords={"T": t})
    spells = app_calc._cumsum_flagged_diff(precip, "T")

    np.testing.assert_array_equal(spells, [1, 2, 0, 0, 5, 0, 0, 0, 0])


def test__cumsum_flagged_diff_all_flagged():

    t = pd.date_range(start="2000-05-01", end="2000-05-10", freq="1D")
    values = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    precip = xr.DataArray(values, coords={"T": t})
    spells = app_calc._cumsum_flagged_diff(precip, "T")

    np.testing.assert_array_equal(spells, [10, 0, 0, 0, 0, 0, 0, 0, 0])


def test__cumsum_flagged_diff_all_unflagged():

    t = pd.date_range(start="2000-05-01", end="2000-05-10", freq="1D")
    values = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    precip = xr.DataArray(values, coords={"T": t})
    spells = app_calc._cumsum_flagged_diff(precip, "T")

    np.testing.assert_array_equal(spells, [0, 0, 0, 0, 0, 0, 0, 0, 0])


def test__cumsum_flagged_diff_coord_size_1():

    t = pd.date_range(start="2000-05-01", end="2000-05-10", freq="1D")
    values = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 0])
    precip = xr.DataArray(values, coords={"T": t})
    precip_0_1 = precip[0]
    precip_0_0 = precip_0_1 - 1
    spells1 = app_calc._cumsum_flagged_diff(precip_0_1, "T")
    spells0 = app_calc._cumsum_flagged_diff(precip_0_0, "T")

    assert spells1 == 1
    assert spells0 == 0


def test_spells_length():

    t = pd.date_range(start="2000-05-01", end="2000-05-10", freq="1D")
    values = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0])
    precip = xr.DataArray(values, coords={"T": t})
    spells = app_calc.spells_length(precip, "T")
    expected = [1, np.nan, np.nan, 2, np.nan, np.nan, np.nan, np.nan, 3, np.nan]

    np.testing.assert_array_equal(spells.isnull(), np.isnan(expected))
    np.testing.assert_array_equal(
        spells.where(~spells.isnull(), other=0),
        np.where(np.isnan(expected), 0, expected)
    )


def test_count_days_in_spells():

    t = pd.date_range(start="2000-05-01", end="2000-05-10", freq="1D")
    values = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0])
    precip = xr.DataArray(values, coords={"T": t})
    days_in_spells = app_calc.count_days_in_spells(precip, "T", min_spell_length=2)

    assert days_in_spells == 5

def test_length_of_longest_spell():

    t = pd.date_range(start="2000-05-01", end="2000-05-10", freq="1D")
    values = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0])
    precip = xr.DataArray(values, coords={"T": t})
    longest_spell = app_calc.length_of_longest_spell(precip, "T")

    assert longest_spell == 3

def test_mean_length_of_spells():

    t = pd.date_range(start="2000-05-01", end="2000-05-10", freq="1D")
    values = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0])
    precip = xr.DataArray(values, coords={"T": t})
    mean_spell = app_calc.mean_length_of_spells(precip, "T", min_spell_length=2)

    assert mean_spell == 2.5

def test_median_length_of_spells():

    t = pd.date_range(start="2000-05-01", end="2000-05-10", freq="1D")
    values = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0])
    precip = xr.DataArray(values, coords={"T": t})
    mendian_spell = app_calc.median_length_of_spells(precip, "T")

    assert mendian_spell == 2
