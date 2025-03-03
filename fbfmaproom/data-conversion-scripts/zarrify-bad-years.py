import cftime
import csv
import numpy as np
import pandas as pd
from pathlib import Path


DEFAULT_ROOT = '/data/aaron/fbf-candidate'


def convert(csvfile, target_month, zarrpath):
    # switch from 0-based counting to 1-based counting
    month = int(target_month) + 1

    # using 360 day calendar--all months have 30 days
    day = int(target_month % 1 * 30) + 1

    df = pd.read_csv(csvfile, index_col=0, names=['year', 'rank'])
    assert df.index.dtype == np.int64
    assert df.dtypes['rank'] == np.int64

    df['T'] = [cftime.Datetime360Day(y, month, day) for y in df.index]
    df = df.set_index('T')
    da = df.to_xarray()
    print(da)
    da.to_zarr(zarrpath)


HELP_TEXT ="""Convert bad years data from CSV to zarr.  Expected CSV format has
 two-columns: the first column is a four-digit year, and the second
 column is a rank number (smaller numbers are more bad). Ties in rank
 are allowed, with the caveat that results will be weird if there are
 ties within the worst n %, where n is the value of the
 slider. Missing values are not allowed--if a year hasn't been
 considered for ranking, omit it from the file. Years that are known
 not to be among the worst but haven't been ranked among themselves
 can be indicated as tied for an arbitrary rank, e.g. 100.  """


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=HELP_TEXT)
    parser.add_argument('--datadir', default=DEFAULT_ROOT)
    parser.add_argument('dataset_name')
    parser.add_argument(
        'target_month',
        type=float,
        help='Season center in months since Jan 1, as in config file. 0.0 <= target_month <= 12.0'
    )
    args = parser.parse_args()

    datadir = Path(args.datadir)
    infile = datadir / f'original-data/{args.dataset_name}.csv'
    outdir = datadir / f'{args.dataset_name}.zarr'
    convert(infile, args.target_month, outdir)
