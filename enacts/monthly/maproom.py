"""
`maproom.py` defines functions that generate content dynamically in response to selections made by the user.
It can be run from the command line to test the application during development.
"""
# Import libraries used
import dash
from dash import html
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
import flask
import json
import os

import psycopg2
from psycopg2 import sql
import shapely
from shapely import wkb
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely import geometry

import plotly.express as px

import pingrid

from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import math

import urllib
import xarray as xr

from . import layout

GLOBAL_CONFIG = pingrid.load_config(os.environ["CONFIG"])
CONFIG = GLOBAL_CONFIG["monthly"]

DATA_DIR = GLOBAL_CONFIG["data_dir"] # Path to data
PREFIX = CONFIG["prefix"] # Prefix used at the end of the maproom url
TILE_PFX = "/tile"

with psycopg2.connect(**GLOBAL_CONFIG["db"]) as conn:
    s = sql.Composed([sql.SQL(GLOBAL_CONFIG['shapes_adm'][0]['sql'])])
    df = pd.read_sql(s, conn)
    clip_shape = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))[0]

def read_data(name):
    data = xr.open_dataarray(f"{DATA_DIR}/{name}.zarr", engine="zarr")
    return data

SERVER = flask.Flask(__name__)
APP = dash.Dash(
    __name__,
    server=SERVER,
    #url_base_pathname=f"{PREFIX}/",
    requests_pathname_prefix=f"/python_maproom{PREFIX}/",
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        # "https://use.fontawesome.com/releases/v5.12.1/css/all.css",
    ],
)

APP.title = CONFIG["map_title"]
APP.layout = layout.layout() # Calling the layout function in `layout.py` which includes the layout definitions.

@APP.callback( # Callback to return the raster layer of the map
    Output("map_raster", "url"),
    Input("variable", "value"),
    Input("mon0", "value"),
)
def update_map(variable, month):
    var = CONFIG["vars"][variable]

    mon = { "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
            "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
            "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12 }[month]

    qstr = urllib.parse.urlencode({
        "variable": variable,
        "month": mon,
    })
    #return ""
    return f"/python_maproom/monthly-climatology/tile/{{z}}/{{x}}/{{y}}?{qstr}"


@APP.callback( # Callback for updating the location of the market on the map.
    Output("loc_marker","position"),
    Input("map","click_lat_lng"),
)
def pick_location(click_lat_lng):
    if click_lat_lng == None:
        return GLOBAL_CONFIG['map_center']
    return click_lat_lng                           #  in the data to where the user clicked on the map.

def read_data(name):
    data = xr.open_dataarray(f"{DATA_DIR}/{name}.zarr", engine="zarr")
    return data

@APP.callback(
    Output("plot","figure"),
    Input("loc_marker","position"),
    Input("variable","value")
)
def create_plot(marker_loc, variable): # Callback that creates bar plot to display data at a given point.
    var = CONFIG["vars"][variable]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    try:
        DATA = read_data(var['id'])
        data = pingrid.sel_snap(DATA,marker_loc[0], marker_loc[1])
        base = data.resample(T="1M")
        if var['id'] == "rfe":
            base = base.sum()
        else:
            base = base.mean()
        base = base.groupby("T.month")
        # avg = base.mean()
        #  #.load().resample(T="1M").mean().groupby("T.month").

        # q95 = quantile(0.95)
        # q50 = quantile(0.50)
        # q95 = quantile(0.05)
    except KeyError:
        return pingrid.error_fig(error_msg="Data missing at this location.") # Error fig if marker is out of bounds of data.
 
    # bar_plot = px.bar( # Create the bar plot using plotly express
    #     clim, x=months, y=clim,
    #     title = f"{variable} monthly climatology",
    #     labels = {"x": "Time (months)", "y": f"{variable} ({DATA.attrs['units']})"},
    # )
    return {
        'data': [
            {'x': months, 'y': base.mean().values, 'type': 'bar', 'name': 'average'},

            {'x': months, 'y': base.quantile(0.95).values, 'type': 'scatter', 'name': '95%-ile'},

            {'x': months, 'y': base.quantile(0.50).values, 'type': 'scatter', 'name': '50%-ile'},

            {'x': months, 'y': base.quantile(0.05).values, 'type': 'scatter', 'name': '5%-ile'},
        ],
        'layout': {
            'title': f"{variable} monthly climatology",
            'labels': {
                "x": "Month",
                "y": f"{variable} ({DATA.attrs['units']})"
            },
        },
    }


@APP.callback(
    Output("map_colorbar", "colorscale"),
    Output("map_colorbar", "min"),
    Output("map_colorbar", "max"),
    Input("variable", "value"),
)
def set_colorbar(variable): #setting the color bar colors and values
    var = CONFIG["vars"][variable]
    colormap = select_colormap(var['id'])
    return (
        pingrid.to_dash_colorscale(colormap),
        var['min'],
        var['max'],
    )


def select_colormap(var):
    rain = pingrid.RAINFALL_COLORMAP
    temp = pingrid.RAINBOW_COLORMAP
    if var == "rfe":
        return rain
    elif var == "tmin":
        return temp
    elif var == "tmax":
        return temp
    elif var == "tmean":
        return temp

@SERVER.route(f"/tile/<int:tz>/<int:tx>/<int:ty>")
def tile(tz, tx, ty):
    parse_arg = pingrid.parse_arg
    var = parse_arg("variable")
    month = parse_arg("month", int)

    x_min = pingrid.tile_left(tx, tz)
    x_max = pingrid.tile_left(tx + 1, tz)
    # row numbers increase as latitude decreases
    y_max = pingrid.tile_top_mercator(ty, tz)
    y_min = pingrid.tile_top_mercator(ty + 1, tz)

    varobj = CONFIG['vars'][var]
    data = read_data(varobj['id'])

    if (
            x_min > data['X'].max() or
            x_max < data['X'].min() or
            y_min > data['Y'].max() or
            y_max < data['Y'].min()
    ):
        return pingrid.empty_tile()

    def clip(x):
        res = x['X'][1].item() - x['X'][0].item()
        return x.sel(
            X=slice(x_min - x_min % res, x_max + res - x_max % res),
            Y=slice(y_min - y_min % res, y_max + res - y_max % res),
            T=x['T'].dt.month == month,
        )
        
    tile = clip(data)

    groups = tile.groupby('T.year')
    if varobj['id'] == "rfe":
        tile = groups.sum('T')
    else:
        tile = groups.mean('T')

    tile = tile.mean('year')

    colormap = select_colormap(varobj['id'])

    tile = tile.rename({'X': "lon", 'Y': "lat"})

    tile.attrs["colormap"] = colormap
    tile.attrs["scale_min"] = varobj['min']
    tile.attrs["scale_max"] = varobj['max']

    result = pingrid.tile(tile, tx, ty, tz, clip_shape)


    return result

@SERVER.route(f"/python_maproom{PREFIX}/health")
def health_endpoint():
    return flask.jsonify({'status': 'healthy', 'name': 'python_maproom'})


if __name__ == "__main__":
    APP.run_server(
        debug=GLOBAL_CONFIG["mode"] != "prod"
    )
