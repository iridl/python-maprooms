import calendar
import cftime
import cptio
import numpy as np
from pathlib import Path
import scipy.stats
import xarray as xr
import zarr


DEFAULT_ROOT = '/data/aaron/fbf-candidate'

def sqrt(x):
    return xr.apply_ufunc(np.sqrt, x)

def tcdf(x, dof, loc=0, scale=1):
    return xr.apply_ufunc(scipy.stats.t.cdf, x, dof, loc, scale)

def calc_pne(obs, hindcasts, forecasts, dof=None, quantile_first_year=None, quantile_last_year=None):
    if dof is None:
        dof = len(hindcasts['T']) - 1
    if quantile_first_year is None:
        quantile_first_year = hindcasts['T'].dt.year[0].item()
    if quantile_last_year is None:
        quantile_last_year = hindcasts['T'].dt.year[-1].item()

    assert len(obs.groupby('T.month')) == 1
    assert len(hindcasts.groupby('T.month')) == 1
    assert len(forecasts['T']) == 0 or len(forecasts.groupby('T.month')) == 1

    # np.arange(.05, 1, .05) yields quantiles that are slightly off, e.g. .9500000001
    quantiles = [.05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95]

    thresholds = (
        obs
        .sel(T=slice(f'{quantile_first_year}-01-01', f'{quantile_last_year}-12-31'))
        .quantile(quantiles, 'T')
    )

    hindcast_pne = tcdf(thresholds, dof, hindcasts['mu'], sqrt(hindcasts['var']))
    forecast_pne = tcdf(thresholds, dof, forecasts['mu'], sqrt(forecasts['var']))
    concat_pne = xr.concat([hindcast_pne, forecast_pne], 'T')
    concat_pne = concat_pne.rename(obs='pne')

    return concat_pne

abbrevs = [None, 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def read_v2_one_issue_month(path):
    hindcasts = xr.Dataset(
        dict(
            mu=open_one(path / 'MME_deterministic_hindcasts.nc'),
            var=open_one(path / 'MME_hindcast_prediction_error_variance.nc'),
        ),
    )
    mu_files = list(path.glob('MME_deterministic_forecast_*.nc'))
    var_files = list(path.glob('MME_forecast_prediction_error_variance_*.nc'))
    if mu_files and var_files:
        mu_da = open_multi(mu_files).load()
        var_da = open_multi(var_files).load()
        forecasts = xr.Dataset(
            dict(mu=mu_da, var=var_da)
        )
    else:
        forecasts = hindcasts.isel(T=slice(0, 0))
    return hindcasts, forecasts,

def to_360_date(year, month, day):
    isleap = calendar.isleap(year)
    if day == 1:
        newday = 1
    elif day in (14, 15, 16, 17): # todo be as pedantic as for the last day?
        newday = 16
    elif (
            day == 28 and month == 2 and not isleap or
            day == 29 and month == 2 and isleap or
            day == 30 and month in (2, 4, 6, 9, 11) or
            day == 31 and month in (1, 3, 5, 7, 8, 10, 12)
    ):
        newday = 30
    else:
        assert False, f"Bad date {year}-{month}-{day}"

    return cftime.Datetime360Day(year, month, newday)

def to_360_coord(c):
    return [
        to_360_date(year, month, day)
        for year, month, day in zip(
                c.dt.year.values,
                c.dt.month.values,
                c.dt.day.values
        )
    ]
                
def to_360_ds(ds):
    ds['S'] = to_360_coord(ds['S'])
    return ds

def load_pne(path):
    if (path / 'obs.nc').is_file():
        obs_da = open_one(path / 'obs.nc')
    else:
        # for backwards compatibility with some old forecast datasets,
        # try again with tsv
        obs_da = next(iter(cptio.open_cptdataset(path / 'obs.tsv').data_vars.values()))
    obs = xr.Dataset(dict(
        obs=obs_da
    ))
    pne_per_issue_month = []
    for monthno in range(1, 13):
        month_path = path / f'{monthno:02}'
        if month_path.exists():
            hindcasts, forecasts = read_v2_one_issue_month(month_path)
            pne = calc_pne(obs, hindcasts, forecasts)
            pne = pne.swap_dims(T='S')
            pne = pne.reset_coords(drop=True)
            pne = to_360_ds(pne)
            pne_per_issue_month.append(pne)
    return xr.merge(pne_per_issue_month, compat='no_conflicts')


def open_one(path):
    da = xr.open_dataarray(path)
    return da

def open_multi(paths):
    da = next(iter(xr.open_mfdataset(paths).data_vars.values()))
    return da


def zarrify(path, datadir):
    print(path)
    pne = load_pne(datadir / 'original-data' / path)
    pne['quantile'] = (pne['quantile'] * 100).astype(int)
    pne['pne'] = pne['pne'] * 100
    # Some input datasets are in decreasing latitude order, which
    # makes some geometry calculations fail.
    pne = pne.sortby('Y')
    print(pne)
    abspath = datadir / f'{path}.zarr'
    try:
        pne.to_zarr(abspath)
    except zarr.errors.ContainsGroupError:
        raise Exception(f'{abspath} already exists. Remove it first if you want to replace it.')


if __name__ == '__main__':
    import argparse
    import os

    os.umask(0o002)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name')
    parser.add_argument('--datadir', default=DEFAULT_ROOT)
    args = parser.parse_args()
    zarrify(args.dataset_name, datadir=Path(args.datadir))
