import xarray as xr


#This is what we should need for the app
#For the map:
#-- read_data for histo and 1 model
#-- seasonal_data for specific years range
#-- seasanal_data.mean(dim="T") - seasonal_histo.mean(dim="T")
#-- unit_conversion
#-- then maybe some other things to beautify map tbd
#
#For the ts:
#-- read_data for all histo and scenarios
#-- select X and Y
#-- seasonal_data
#-- append or figure out what the graph will need as input to plot histo
#followed by different scenarios
#-- unit_conversion
#-- then maybe some other things to beautify map tbd


def read_data(scenario, model, variable):

    return xr.open_zarr(
        f'/Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global'
        f'/daily/{scenario}/{model}/zarr/{variable}'
    )[variable]


def seasonal_data(daily_data, season_center, start_year=None, end_year=None):

    if ((end_year != None) and (season_center in [12, 1])):
        end_year = end_year + 1
    season_months = [
        season_center - 1 if season_center > 1 else 12,
        season_center,
        season_center + 1 if season_center < 12 else 1,
    ]
    return (daily_data
        .sel(T=slice(f"{start_year}", f"{end_year}"))
        #.where(lambda x: (
        #    (x["T"].dt.month == season_months[0])
        #    + (x["T"].dt.month == season_months[1])
        #    + (x["T"].dt.month == season_months[2])
        #), drop=True)
        .resample(T="1M").mean()
        .rolling(T=3, center=True).mean().dropna("T")
        .where(lambda x: x["T"].dt.month == season_center, drop=True)
    )


def unit_conversion(variable):
    #if precipitation variable, change from kg to mm per day
    if variable == 'pr':
        variable *= 86400
        variable.attrs['units'] = 'mm/day' #rename unit to converted
    elif variable in ['tas', 'tasmin', 'tasmax']:
        variable -= 273.15 
        variable.attrs['units'] = 'Celsius'
    
    return variable

data = read_data("ssp126", "GFDL-ESM4", "pr")
seas = seasonal_data(data, 2, 2061, 2090)
print(seas)