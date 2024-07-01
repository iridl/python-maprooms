import numpy as np
import pandas as pd
import xarray as xr
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from urllib.request import urlopen
import json

def compute_annual_monthly_avg(ds, variable, start_year=None, end_year=None):
    #compute the avg (variable) for each month across the specified years
    
    #if precipitation variable, change from kg to mm per day
    if variable == 'pr':
        ds[variable] *= 86400
    
    if start_year and end_year:
        ds = ds.sel(T=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    # Convert daily data to monthly
    monthly_ds = ds.resample(T="1M").mean()

    # Compute the average for each month across all years
    monthly_avg = monthly_ds.mean(dim=["X", "Y"])
    annual_monthly_avg = monthly_avg.groupby("T.month").mean(dim="T")
    df = annual_monthly_avg.to_dataframe().reset_index()
    return df

def compute_annual_seasonal_avg(monthly_ds, variable, start_year=None, end_year=None):
    #compute monthly avgs across all years
    monthly_avg = monthly_ds.mean(dim=["X", "Y"])

    # Compute rolling seasonal average (3-month rolling window, center-aligned)
    rolling_avg = monthly_avg.rolling(T=3, center=True).mean()
    
    # Compute the average variable for each rolling season across all years
    rolling_seasonal_avg = rolling_avg.groupby("T.month").mean(dim="T")
    
    # Create a DataFrame for seasonal averages
    seasonal_avgs_df = rolling_seasonal_avg.to_dataframe().reset_index()

    # Assign the season column with correct length
    seasonal_avgs_df['season'] = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
                                   'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']

    return seasonal_avgs_df
    

def write_to_csv(df, scenario, model, variable, output_dir):

    # Write to CSV file
    file_path = f'{output_dir}/{scenario}_{model}_{variable}_annual_monthly_avg.csv'
    df.to_csv(file_path, index=False)


def map_averages(df, model, variable):

    df = df.sort_values(by='month')
    map_fig = px.imshow(
        df.pivot(index='Y', columns='X', values=variable), 
        labels={'color': 'Kelvins'},
        title=f'{variable} Monthly Averages',
        origin='lower'  # Set the origin to lower to flip the y-axis if needed
    )
    return fig

    

def main(scenario, model, variable, start_year=None, end_year=None, output_dir='/home/sz3116/python-maprooms/pepsico/resources'):
    #main to run functions

    # Open zarr file 
    ds = xr.open_zarr(f'//Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/historical/{model}/zarr/{variable}')

    print("\nVariable attributes:")
    for var in ds.data_vars:
        print(f"\nAttributes for variable '{var}':")
        print(ds[var].attrs)
    
    # Compute the average variable for each month across all years
    annual_monthly_avg = compute_annual_monthly_avg(ds, variable, start_year, end_year)
    
    #call seasonal function
    rolling_seasonal_avg = compute_annual_seasonal_avg(ds, variable, start_year, end_year)
    
    # Write the data to a CSV file
    write_to_csv(annual_monthly_avg, scenario, model, variable, output_dir)
    write_to_csv(rolling_seasonal_avg, scenario, model, f'{variable}_rolling', output_dir)

    fig = map_averages(df, variable)
    fig.show()
    
   
    #also print in terminal
    print(annual_monthly_avg)
    print(rolling_seasonal_avg)

    
    
# testing
if __name__ == "__main__":
    scenario = "historical"
    model = "GFDL-ESM4"
    variable = "tasmin"
    start_year = 1950
    end_year = 2014
    output_dir = '/home/sz3116/python-maprooms/pepsico/resources'

    main(scenario, model, variable, start_year, end_year, output_dir)

print("done")

