import numpy as np
import pandas as pd
import xarray as xr
import plotly.express as px

#for tmin variable

#open zarr file 
ds = xr.open_zarr('/Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/historical/GFDL-ESM4/zarr/tasmin')


#convert daily data to monthly
monthly_ds = ds.resample(T="1M").mean()

#spatial avg for each month
monthly_avg_tasmin = monthly_ds.mean(dim=["X", "Y"])
#print(monthly_ds.head())

# Convert to Pandas DataFrame for use with Plotly
df = monthly_avg_tasmin.to_dataframe().reset_index()

#print("averages")
#print(df.head(800))

df.to_csv('/home/sz3116/python-maprooms/pepsico/resources/monthly_tasmin.csv', index=False)
print("done")






'''
# Create the spatial plot
fig = px.imshow(hurs_specific_month.values, 
                labels={'color': 'Relative Humidity'},
                x=monthly_ds['X'].values,
                y=monthly_ds['Y'].values,
                title=f'Relative Humidity for {specific_month}')
fig.update_layout(
    xaxis_title='Longitude',
    yaxis_title='Latitude'
)
fig.show()
'''
'''
#print the entire dataframe
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Monthly averages for hurs:")
    print(df)

# Save to CSV (optional)
'''
