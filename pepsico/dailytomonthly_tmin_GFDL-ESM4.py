import numpy as np
import pandas as pd
import xarray as xr
import plotly.express as px

#for tmin variable, model:GFDL-ESM4

#open zarr file 
ds = xr.open_zarr('/Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/historical/GFDL-ESM4/zarr/tasmin')

# Convert daily data to monthly
monthly_ds = ds.resample(T="1M").mean()

# Compute the average tasmin for each month across all years
monthly_avg_tasmin = monthly_ds.mean(dim=["X", "Y"])
annual_monthly_avg = monthly_avg_tasmin.groupby("T.month").mean(dim="T")

# Prepare the data for the CSV
data = {'month': [], 'tasmin': []}
for month in range(1, 13):
    month_data = annual_monthly_avg.sel(month=month)
    tasmin_value = month_data['tasmin'].values  # Extract the tasmin values
    avg_tasmin_value = np.mean(tasmin_value)  # Calculate the average to get a single value
    data['month'].append(month)
    data['tasmin'].append(avg_tasmin_value)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to a single CSV file
df.to_csv('/home/sz3116/python-maprooms/pepsico/resources/annual_monthly_GFDL-ESM4_tasmin.csv', index=False)

print("done")
