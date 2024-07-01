import numpy as np
import pandas as pd
import xarray as xr
import plotly.express as px

# Open zarr file 
ds = xr.open_zarr('/Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/historical/GFDL-ESM4/zarr/tasmin')

# Convert daily data to monthly
monthly_ds = ds.resample(T="1M").mean()

# Compute the average tasmin for each month across all years
annual_monthly_avg = monthly_ds.groupby("T.month").mean(dim="T")

# Prepare the data for the CSV (too big to print in terminal)
data = {'month': [], 'tasmin': []}
for month in range(1, 13):
    month_data = annual_monthly_avg.sel(month=month)
    tasmin_value = month_data['tasmin'].mean(dim=["X", "Y"]).values  # Get the mean tasmin value
    data['month'].append(month)
    data['tasmin'].append(tasmin_value)

    # Create a DataFrame for the spatial data for this month
    df = month_data['tasmin'].to_dataframe().reset_index()

    # Sort latitudes in ascending order if necessary - for if map is flipped
    df = df.sort_values(by='Y')

    # Create the map
    fig = px.imshow(
        df.pivot(index='Y', columns='X', values='tasmin'), 
        labels={'color': 'Kelvins'},
        title=f'Tasmin for Month {month}',
        origin='lower'  # Set the origin to lower to flip the y-axis
    )
    
    
    # Save the map as an HTML file - access through port 8000
    fig.write_html(f'/home/sz3116/python-maprooms/pepsico/resources/map_month_{month}.html')


print("done")
