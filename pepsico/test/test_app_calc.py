
import numpy as np
import pandas as pd
import xarray as xr
import app_calc

def test_number_extreme_events_within_days():
    t = pd.date_range(start="2000-05-01", end="2000-05-10", freq="1D")
    values = 1 + 0*np.arange(t.size)
    precip = xr.DataArray(values, coords={"T": t})
    count = app_calc.number_extreme_events_within_days(precip, 2, 3)

    np.testing.assert_array_equal(count, [3])
    