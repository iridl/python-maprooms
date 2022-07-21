import os
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
from pathlib import Path
import pyaconf

CONFIG = pyaconf.load(os.environ["CONFIG"])

RR_MRG_NC_PATH = CONFIG["rr_mrg_nc_path"]
RR_MRG_ZARR_PATH = CONFIG["rr_mrg_zarr_path"]
RESOLUTION = CONFIG["rr_mrg_resolution"]
CHUNKS = CONFIG["rr_mrg_chunks"]

#Read daily files of daily rainfall data
#Concatenate them against added time dim made up from filenames

RR_MRG_PATH = Path(RR_MRG_NC_PATH)
RR_MRG_FILE = list(sorted(RR_MRG_PATH.glob("*.nc")))


def set_up_dims(xda):
    datestr = Path(xda.encoding["source"]).name.split("_")[2]
    xda = xda.expand_dims(T = [dt.datetime(
      int(datestr[0:4]),
      int(datestr[4:6]),
      int(datestr[6:8])
    )])
    xda = xda.rename({'Lon': 'X','Lat': 'Y'})
    return xda

rr_mrg = xr.open_mfdataset(
    RR_MRG_FILE,
    preprocess = set_up_dims,
    parallel=False
).precip

if not np.isclose(rr_mrg['X'][1] - rr_mrg['X'][0], RESOLUTION):
    # TODO this method of regridding is inaccurate because it pretends
    # that (X, Y) define a Euclidian space. In reality, grid cells
    # farther from the equator cover less area and thus should be
    # weighted less heavily. Also, consider using conservative
    # interpolation instead of bilinear, since when going to a much
    # coarser resoution, bilinear discards a lot of information. See [1],
    # and look into xESMF.
    #
    # [1] https://climatedataguide.ucar.edu/climate-data-tools-and-analysis/regridding-overview
    rr_mrg = rr_mrg.interp(
        X=np.arange(rr_mrg.X.min(), rr_mrg.X.max() + RESOLUTION, RESOLUTION),
        Y=np.arange(rr_mrg.Y.min(), rr_mrg.Y.max() + RESOLUTION, RESOLUTION),
    )

rr_mrg = rr_mrg.chunk(chunks=CHUNKS)

xr.Dataset().merge(rr_mrg).to_zarr(
  store = RR_MRG_ZARR_PATH
)

