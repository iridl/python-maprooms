import numpy as np
import pandas as pd
import xarray as xr
import plotly.express as px

#for tasmax variable, model:GFDL-ESM4

#open zarr file 
ds = xr.open_zarr('/Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/historical/GFDL-ESM4/zarr/tasmax')

# Convert daily data to monthly
monthly_ds = ds.resample(T="1M").mean()

# Compute the average tasmax for each month across all years
monthly_avg_tasmax = monthly_ds.mean(dim=["X", "Y"])
annual_monthly_avg = monthly_avg_tasmax.groupby("T.month").mean(dim="T")

# Prepare the data for the CSV
data = {'month': [], 'tasmax': []}
for month in range(1, 13):
    month_data = annual_monthly_avg.sel(month=month)
    tasmax_value = month_data['tasmax'].values  # Extract the tasmax values
    avg_tasmax_value = np.mean(tasmax_value)  # Calculate the average to get a single value
    data['month'].append(month)
    data['tasmax'].append(avg_tasmax_value)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to a single CSV file
df.to_csv('/home/sz3116/python-maprooms/pepsico/resources/annual_monthly_GFDL-ESM4_tasmax.csv', index=False)

print("done")