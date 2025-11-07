import argparse
import cftime
import xarray as xr
import os
import shutil

import pingrid


if __name__ == '__main__':
    config = pingrid.load_config(os.environ["CONFIG"])

    os.umask(0o002)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cookiefile', type=os.path.expanduser)
    parser.add_argument(
        '--datadir',
        default=config['data_root'],
        type=os.path.expanduser,
    )
    parser.add_argument('datasets', nargs='*')
    opts = parser.parse_args()

    datasets = {
        ds['path'].removesuffix('.zarr'): ds
        for country_config in config['countries'].values()
        for ds in (
                country_config['datasets']['observations'] |
                country_config['datasets']['forecasts']
        ).values()
        if 'url' in ds
    }

    if opts.datasets:
        datasets = {key: datasets[key] for key in opts.datasets}

    for key, ds in datasets.items():
        name = key
        pattern = ds['url']
        slices = ds.get('url_args', [{}])
        print(name)
        for i, args in enumerate(slices):
            ncfilepath = f'{opts.datadir}/{name}-{i}.nc'
            leafdir = os.path.dirname(ncfilepath)

            if not os.path.exists(leafdir):
                os.makedirs(leafdir)

            if opts.cookiefile is None:
                cookieopt = ""
            else:
                cookieopt = f"-b {opts.cookiefile}"

            url = pattern.format(**args)
            status = os.system(f"curl {cookieopt} -o {ncfilepath} '{url}data.nc'")
            if status != 0:
                raise Exception(f"Download failed: {url}")
            assert os.path.exists(ncfilepath)
        zarrpath = "%s/%s.zarr" % (opts.datadir, name)
        print("Converting to zarr")
        dss = [
            pingrid.open_dataset(f'{opts.datadir}/{name}-{i}.nc')
            for i in range(len(slices))
        ]
        ds = xr.concat(dss, 'T')
        # TODO do this in Ingrid
        if 'Y' in ds and ds['Y'][0] > ds['Y'][1]:
            ds = ds.reindex(Y=ds['Y'][::-1])
        if 'P' in ds:
            ds = ds.chunk({'P': 1})
        if os.path.exists(zarrpath):
            shutil.rmtree(zarrpath)
        ds.to_zarr(zarrpath)
