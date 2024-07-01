import numpy as np
import pandas as pd
import xarray as xr
import plotly.express as px

# Open zarr file 
ds = xr.open_zarr('/Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/historical/GFDL-ESM4/zarr/tasmax')

# Convert daily data to monthly
monthly_ds = ds.resample(T="1M").mean()

# Compute the average tasmax for each month across all years
annual_monthly_avg = monthly_ds.groupby("T.month").mean(dim="T")

# Prepare the data for the CSV (too big to print in terminal)
data = {'month': [], 'tasmax': []}
for month in range(1, 13):
    month_data = annual_monthly_avg.sel(month=month)
    tasmax_value = month_data['tasmax'].mean(dim=["X", "Y"]).values  # Get the mean tasmax value
    data['month'].append(month)
    data['tasmax'].append(tasmax_value)

    # Create a DataFrame for the spatial data for this month
    df = month_data['tasmax'].to_dataframe().reset_index()

    # Sort latitudes in ascending order if necessary - for if map is flipped
    df = df.sort_values(by='Y')

    # Create the map
    fig = px.imshow(
        df.pivot(index='Y', columns='X', values='tasmax'), 
        labels={'color': 'Kelvins'},
        title=f'Tasmax for Month {month}',
        origin='lower'  # Set the origin to lower to flip the y-axis
    )
    
    
    # Save the map as an HTML file - access through port 8000
    fig.write_html(f'/home/sz3116/python-maprooms/pepsico/resources/GFDL-ESM4/tmax_map_month_{month}.html')


print("done")