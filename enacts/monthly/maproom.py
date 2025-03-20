"""
`maproom.py` defines functions that generate content dynamically in response to selections made by the user.
It can be run from the command line to test the application during development.
"""
# Import libraries used
import dash
from dash import html
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
import dash_leaflet as dlf
import dash_leaflet.express as dlx
import json

import psycopg2
from psycopg2 import sql
import shapely
from shapely import wkb
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely import geometry

import plotly.express as px

import pingrid
from pingrid import CMAPS

from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import math

import urllib
import xarray as xr

import calc

from . import layout
from globals_ import FLASK, GLOBAL_CONFIG

CONFIG = GLOBAL_CONFIG["maprooms"]["monthly"]

READ_PARAMS = {"ds_conf": GLOBAL_CONFIG['datasets']}

def register(FLASK, config):
    # Prefix used at the end of the maproom url
    PREFIX = f'{GLOBAL_CONFIG["url_path_prefix"]}/{config["core_path"]}'
    TILE_PFX = f"{PREFIX}/tile"

    APP = dash.Dash(
        __name__,
        server=FLASK,
        url_base_pathname=f"{PREFIX}/",
        external_stylesheets=[dbc.themes.BOOTSTRAP,],
    )

    APP.title = config["title"]
    # Calling the layout function in `layout.py`
    # which includes the layout definitions.
    APP.layout = layout.layout()

    def get_shapes(query):
        with psycopg2.connect(**GLOBAL_CONFIG["db"]) as conn:
            s = sql.SQL(query)
            df = pd.read_sql(s, conn)

        df["the_geom"] = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))
        df["the_geom"] = df["the_geom"].apply(
            # lambda x: ((x if isinstance(x, MultiPolygon) else MultiPolygon([x]))
            lambda x: (x
                   .simplify(0.01, preserve_topology=False))
        )
        shapes = df["the_geom"].apply(shapely.geometry.mapping)
        for i in df.index: #this adds the district layer as a label in the dict
            shapes[i]['label'] = df['label'][i]
        return {"features": shapes}


    @APP.callback(
        Output("app_title", "children"),
        Output("map", "center"),
        Input("location", "pathname"),
    )
    def initialize(path):
        rr_mrg = calc.read_enacts(variable="precip", **READ_PARAMS)
        center_of_the_map = [
            ((rr_mrg["Y"][int(rr_mrg["Y"].size/2)].values)),
            ((rr_mrg["X"][int(rr_mrg["X"].size/2)].values)),
        ]
        return config["title"], center_of_the_map


    @APP.callback( # Callback to return the raster layer of the map
        Output("map_layers_control", "children"),
        Input("variable", "value"),
        Input("mon0", "value"),
    )
    def update_map(variable, month):
        SHAP = {
            level['name']: get_shapes(level['sql'])
            for level in GLOBAL_CONFIG['datasets']['shapes_adm']
        }
        var = config["vars"][variable]

        mon = { "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
                "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
                "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12 }[month]

        qstr = urllib.parse.urlencode({
            "variable": variable,
            "month": mon,
        })
        return [
            dlf.BaseLayer(
                dlf.TileLayer(
                    opacity=0.6,
                    # Cartodb street map.
                    url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
                ),
                name="Street",
                checked=True,
            ),
            dlf.BaseLayer(
                dlf.TileLayer(
                    opacity=0.6,
                    # opentopomap topography map.
                    url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
                ),
                name="Topo",
                checked=False,
            ),
        ] + [
            dlf.Overlay(
                dlf.GeoJSON(
                    data=v,
                    options={
                        "fill": True,
                        "color": "black",
                        "weight": .5,
                        "fillOpacity": 0,
                    },
                ),
                name=k,
                checked=True,
            )
            for k, v in SHAP.items()
        ] + [
            dlf.Overlay(
                dlf.TileLayer(
                    url=f"{TILE_PFX}/{{z}}/{{x}}/{{y}}?{qstr}",
                    opacity=1,
                    id="map_raster",
                ),
                name="Raster",
                checked=True,
            ),
        ]


    @APP.callback( # Callback for updating the location of the marker on the map.
        Output("loc_marker","position"),
        Input("map","click_lat_lng"),
    )
    def pick_location(click_lat_lng):
        if click_lat_lng == None:
            rr_mrg = calc.read_enacts(variable="precip", **READ_PARAMS)
            return [
                ((rr_mrg["Y"][int(rr_mrg["Y"].size/2)].values)),
                ((rr_mrg["X"][int(rr_mrg["X"].size/2)].values)),
            ]
        return click_lat_lng

    @APP.callback(
        Output("plot","figure"),
        Input("loc_marker","position"),
        Input("variable","value")
    )
    def create_plot(marker_loc, variable):
        # Callback that creates bar plot to display data at a given point.
        var = config["vars"][variable]
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        try:
            DATA = calc.read_enacts(variable=var['id'], **READ_PARAMS)
            data = pingrid.sel_snap(DATA, marker_loc[0], marker_loc[1])
            base = data.resample(T="1M")
            if var['id'] == "precip":
                base = base.sum()
            else:
                base = base.mean()
            base = base.chunk(dict(T=-1)).groupby("T.month")
            # avg = base.mean()
            #  #.load().resample(T="1M").mean().groupby("T.month").

            # q95 = quantile(0.95)
            # q50 = quantile(0.50)
            # q95 = quantile(0.05)
        except KeyError:
            # Error fig if marker is out of bounds of data.
            return pingrid.error_fig(error_msg="Data missing at this location.")
 
        # bar_plot = px.bar( # Create the bar plot using plotly express
        #     clim, x=months, y=clim,
        #     title = f"{variable} monthly climatology",
        #     labels = {"x": "Time (months)", "y": f"{variable} ({DATA.attrs['units']})"},
        # )
        return {
            'data': [
                {
                    'x': months,
                    'y': base.mean().values,
                    'type': 'bar',
                    'name': 'average',
                },
                {
                    'x': months,
                    'y': base.quantile(0.95).values,
                    'type': 'scatter',
                    'name': '95%-ile',
                },
                {
                    'x': months,
                    'y': base.quantile(0.50).values,
                    'type': 'scatter',
                    'name': '50%-ile',
                },
                {
                    'x': months,
                    'y': base.quantile(0.05).values,
                    'type': 'scatter',
                    'name': '5%-ile',
                },
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
        Output("map_colorbar", "children"),
        Input("variable", "value"),
    )
    def set_colorbar(variable): #setting the color bar colors and values
        var = config["vars"][variable]
        colormap = select_colormap(var['id'])
        return dlf.Colorbar(
            id="colorbar",
            colorscale=colormap.to_dash_leaflet(),
            min=var['min'],
            max=var['max'],
            position="bottomleft",
            width=300,
            height=10,
            opacity=1,
            className="p-1",
            style={
                "background": "white",
                "border-style": "inset",
                "-moz-border-radius": "4px",
                "border-radius": "4px",
                "border-color": "LightGrey",
            },
        )


    def select_colormap(var):
        rain = CMAPS["precip"]
        temp = CMAPS["rainbow"]
        if var == "precip":
            return rain
        elif var == "tmax":
            return temp
        elif var == "tmin":
            return temp
        elif var == "tmean":
            return temp

    @FLASK.route(f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>")
    def tile(tz, tx, ty):
        parse_arg = pingrid.parse_arg
        var = parse_arg("variable")
        month = parse_arg("month", int)

        x_min = pingrid.tile_left(tx, tz)
        x_max = pingrid.tile_left(tx + 1, tz)
        # row numbers increase as latitude decreases
        y_max = pingrid.tile_top_mercator(ty, tz)
        y_min = pingrid.tile_top_mercator(ty + 1, tz)

        varobj = config['vars'][var]
        data = calc.read_enacts(variable=varobj['id'], **READ_PARAMS)
    
        if (
            x_min > data['X'].max() or
            x_max < data['X'].min() or
            y_min > data['Y'].max() or
            y_max < data['Y'].min()
        ):
            return pingrid.image_resp(pingrid.empty_tile())

        def clip(x):
            res = x['X'][1].item() - x['X'][0].item()
            return x.sel(
                X=slice(x_min - x_min % res, x_max + res - x_max % res),
                Y=slice(y_min - y_min % res, y_max + res - y_max % res),
                T=x['T'].dt.month == month,
            )
        
        tile = clip(data)

        groups = tile.groupby('T.year')
        if var == "Rainfall":
            tile = groups.sum('T')
        else:
            tile = groups.mean('T')

        tile = tile.mean('year')

        colormap = select_colormap(varobj['id'])
    
        tile = tile.rename({'X': "lon", 'Y': "lat"})

        tile.attrs["colormap"] = colormap
        tile.attrs["scale_min"] = varobj['min']
        tile.attrs["scale_max"] = varobj['max']
    
        with psycopg2.connect(**GLOBAL_CONFIG["db"]) as conn:
            s = sql.Composed(
                [sql.SQL(GLOBAL_CONFIG['datasets']['shapes_adm'][0]['sql'])]
            )
            df = pd.read_sql(s, conn)
            clip_shape = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))[0]
        
        result = pingrid.tile(tile, tx, ty, tz, clip_shape)

        return result
