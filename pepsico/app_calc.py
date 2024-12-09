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
        f'/monthly/{scenario}/{model}/zarr/{variable}'
    )[variable]


def seasonal_data(monthly_data, start_month, end_month, start_year=None, end_year=None):

    #NDJ and DJF are considered part of the year of the 1st month
    if ((end_year != None) and (start_month > end_month)):
        end_year = str(int(end_year) + 1)
    #Reduce data size
    monthly_data = monthly_data.sel(T=slice(start_year, end_year))
    #Find edges of seasons
    start_edges = monthly_data["T"].where(
        lambda x: x["T"].dt.month == start_month, drop=True,
    )
    end_edges = monthly_data["T"].where(
        lambda x: x["T"].dt.month == end_month, drop=True,
    )
    #Select data and edges to avoid partial seasons at the edges of the edges
    monthly_data = monthly_data.sel(T=slice(start_edges[0], end_edges[-1]))
    start_edges = start_edges.sel(T=slice(start_edges[0], end_edges[-1]))
    end_edges = (end_edges
        .sel(T=slice(start_edges[0], end_edges[-1]))
        .assign_coords(T=start_edges["T"])
    )
    #Reduce data size to months in seasons of interest
    months_in_season = (
        (monthly_data["T"] >= start_edges.rename({"T": "group"}))
        & (monthly_data["T"] <= end_edges.rename({"T": "group"}))
    ).sum(dim="group")
    monthly_data = monthly_data.where(months_in_season == 1, drop=True)
    #Create groups of months belonging to same season-year
    seasons_groups = (monthly_data["T"].dt.month == start_month).cumsum() - 1
    #and identified by seasons_starts
    seasons_starts = (
        start_edges.rename({"T": "toto"})[seasons_groups]
        .drop_vars("toto")
        .rename("seasons_starts")
    )

    return (monthly_data
        #Seasonal averages
        .groupby(seasons_starts).mean()
        #Use T as standard name for time dim
        .rename({"seasons_starts": "T"})
        #add seasons_starts/-ends as coords
        .assign_coords(seasons_ends=end_edges)
        .assign_coords(seasons_starts=seasons_starts)
    )


def unit_conversion(variable):
    #if precipitation variable, change from kg to mm per day
    if variable.name == 'pr':
        variable *= 86400
        variable.attrs['units'] = 'mm/day' #rename unit to converted
    elif variable.name in ['tas', 'tasmin', 'tasmax']:
        variable.name -= 273.15 
        variable.attrs['units'] = 'Celsius'
    return variable
