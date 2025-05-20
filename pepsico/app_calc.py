import xarray as xr
import numpy as np
import pandas as pd


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


def read_data(scenario, model, variable, region, unit_convert=False):
    if region == "US-CA":
        xslice = slice(-154, -45)
        yslice = slice(60, 15)
    elif region == "SAMER":
        xslice = slice(-86, -34)
        yslice = slice(16, -60)
    elif region == "SASIA":
        xslice = slice(59, 94)
        yslice = slice(42, 7)
    elif region == "Thailand":
        xslice = slice(85, 115)
        yslice = slice(28, 2)
    data = xr.open_zarr(
        f'/Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global'
        f'/monthly/{scenario}/{model}/zarr/{variable}'
    )[variable].sel(X=xslice, Y=yslice)
    if unit_convert :
        data = unit_conversion(data)
    return data


def seasonal_data(monthly_data, start_month, end_month, start_year=None, end_year=None):

    #NDJ and DJF are considered part of the year of the 1st month
    if ((end_year != None) and (start_month > end_month)):
        end_year = end_year + 1
    #Reduce data size
    if start_year != None :
        start_year = str(start_year)
    if end_year != None :
        end_year = str(end_year)
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


def groupby_seasons(
    daily_data, start_day, start_month, end_day, end_month, time_dim="T",
):
    edges_base = pd.date_range(
        start=daily_data[time_dim][0].dt.floor("D").values,
        # need one more day since right is excluded
        end=(daily_data[time_dim][-1] + np.timedelta64(1, "D")).dt.floor("D").values,
        freq="1D",
    )
    if start_day == 29 and start_month == 2 :
        # don't allow start on 29 Feb: this is pushy
        start_day = 1
        start_month = 3
    offset = 1
    if end_day == 29 and end_month == 2 :
        end_day = 1
        end_month = "Mar"
        offset = 0
    bins = edges_base.where(
        (
            (edges_base.day == start_day)
            & (edges_base.month == start_month)
        )
        | ( # group end is inclusive
            ((edges_base - pd.Timedelta(offset, "D")).day == end_day)
            & ((edges_base - pd.Timedelta(offset, "D")).month == end_month)
        )
    ).dropna()
    return daily_data.groupby_bins(daily_data[time_dim], bins, right=False)


def unit_conversion(variable):
    #if precipitation variable, change from kg to mm per day
    if variable.name == 'pr':
        variable *= 86400
        variable.attrs['units'] = 'mm/day' #rename unit to converted
    elif variable.name in ['tas', 'tasmin', 'tasmax']:
        variable -= 273.15 
        variable.attrs['units'] = '˚C'
    elif variable.name == 'prsn':
        #this is really needed by the colorscale ticks
        variable *= 10e6
        variable.attrs['units'] = f'10e-6 {variable.attrs["units"]}'

    return variable


# This is in enacts calc.py
def strftimeb2int(strftimeb):
    """Convert month values to integers (1-12) from strings.
 
    Parameters
    ----------
    strftimeb : str
        String value representing months of year.               
    Returns
    -------
    strftimebint : int
        Integer value corresponding to month.
    See Also
    --------
    Notes
    -----
    Examples
    --------
    """
    strftimeb_all = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    strftimebint = strftimeb_all[strftimeb]
    return strftimebint
