import os
import flask
import dash
from dash import dcc
from dash import html
from dash import ALL
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import dash_leaflet as dlf
from pathlib import Path
import pingrid 
import layout_crop_suit
import calc
import plotly.graph_objects as pgo
import plotly.express as px
import pandas as pd
import numpy as np
import urllib
import math
import psycopg2
from psycopg2 import sql
import shapely
from shapely import wkb
from shapely.geometry.multipolygon import MultiPolygon
import datetime

GLOBAL_CONFIG = pingrid.load_config(os.environ["CONFIG"])
CONFIG = GLOBAL_CONFIG["crop_suit"]

PFX = CONFIG["core_path"]
TILE_PFX = "/tile"

with psycopg2.connect(**GLOBAL_CONFIG["db"]) as conn:
    s = sql.Composed([sql.SQL(GLOBAL_CONFIG['shapes_adm'][0]['sql'])])
    df = pd.read_sql(s, conn)
    clip_shape = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))[0]

# Reads daily data

DR_PATH = GLOBAL_CONFIG["rr_mrg_zarr_path"]
RR_MRG_ZARR = Path(DR_PATH)
rr_mrg = calc.read_zarr_data(RR_MRG_ZARR)

# Assumes that grid spacing is regular and cells are square. When we
# generalize this, don't make those assumptions.
RESOLUTION = rr_mrg['X'][1].item() - rr_mrg['X'][0].item()
# The longest possible distance between a point and the center of the
# grid cell containing that point.
SERVER = flask.Flask(__name__)
APP = dash.Dash(
    __name__,
    server=SERVER,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
    ],
    url_base_pathname=f"{PFX}/",
    meta_tags=[
        {"name": "description", "content": "Crop Suitability Maproom"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)
APP.title = CONFIG["app_title"]

APP.layout = layout_crop_suit.app_layout()


def adm_borders(shapes):
    with psycopg2.connect(**GLOBAL_CONFIG["db"]) as conn:
        s = sql.Composed(
            [
                sql.SQL("with g as ("),
                sql.SQL(shapes),
                sql.SQL(
                    """
                    )
                    select
                        g.label, g.key, g.the_geom
                    from g
                    """
                ),
            ]
        )
        df = pd.read_sql(s, conn)

    df["the_geom"] = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))
    df["the_geom"] = df["the_geom"].apply(
        lambda x: x if isinstance(x, MultiPolygon) else MultiPolygon([x])
    )
    shapes = df["the_geom"].apply(shapely.geometry.mapping)
    for i in df.index: #this adds the district layer as a label in the dict
        shapes[i]['label'] = df['label'][i]
    return {"features": shapes}


def make_adm_overlay(adm_name, adm_sql, adm_color, adm_lev, adm_weight, is_checked=False):
    border_id = {"type": "borders_adm", "index": adm_lev}
    return dlf.Overlay(
        dlf.GeoJSON(
            id=border_id,
            data=adm_borders(adm_sql),
            options={
                "fill": True,
                "color": adm_color,
                "weight": adm_weight,
                "fillOpacity": 0
            },
        ),
        name=adm_name,
        checked=is_checked,
    )

@APP.callback(
    Output("layers_control", "children"),
    Input("map_choice", "value"),
    Input("season_choice", "value"),
)
def make_map(
        map_choice,
        season_choice,
):
    qstr = urllib.parse.urlencode({
        "map_choice": map_choice,
        "season_choice": season_choice,
    })
    return [
        dlf.BaseLayer(
            dlf.TileLayer(
                url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
            ),
            name="Street",
            checked=False,
        ),
        dlf.BaseLayer(
            dlf.TileLayer(
                url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
            ),
            name="Topo",
            checked=True,
        ),
    ] + [
        make_adm_overlay(
            adm["name"],
            adm["sql"],
            adm["color"],
            i+1,
            len(GLOBAL_CONFIG["shapes_adm"])-i,
            is_checked=adm["is_checked"]
        )
        for i, adm in enumerate(GLOBAL_CONFIG["shapes_adm"])
    ] + [
        dlf.Overlay(
            dlf.TileLayer(
                url=f"{TILE_PFX}/{{z}}/{{x}}/{{y}}?{qstr}",
                opacity=1,
            ),
            name="Crop Suitability",
            checked=True,
        ),
    ]


@APP.callback(
    Output("navbar-collapse", "is_open"),
    Input("navbar-toggler", "n_clicks"),
    State("navbar-collapse", "is_open"),
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


def round_latLng(coord):
    value = float(coord)
    value = round(value, 4)
    return value


@APP.callback(
    Output("hover_feature_label", "children"),
    Input({"type": "borders_adm", "index": ALL}, "hover_feature")
)
def write_hover_adm_label(adm_loc):
    location_description = "the map will return location name"
    for i, adm in enumerate(adm_loc):
        if adm is not None:
            location_description = adm['geometry']['label']
    return f'Mousing over {location_description}'


@APP.callback(
    Output("map_description", "children"),
    Input("map_choice", "value"),
)
def write_map_description(map_choice):
    return CONFIG["map_text"][map_choice]["description"]    

@APP.callback(
    Output("loc_marker", "position"),
    Output("lat_input", "value"),
    Output("lng_input", "value"),
    Input("submit_lat_lng","n_clicks"),
    Input("map", "click_lat_lng"),
    State("lat_input", "value"),
    State("lng_input", "value")
)
def pick_location(n_clicks, click_lat_lng, latitude, longitude):
    if dash.ctx.triggered_id == None:
        lat = rr_mrg.precip["Y"][int(rr_mrg.precip["Y"].size/2)].values
        lng = rr_mrg.precip["X"][int(rr_mrg.precip["X"].size/2)].values
    else:
        if dash.ctx.triggered_id == "map":
            lat = click_lat_lng[0]
            lng = click_lat_lng[1]
        else:
            lat = latitude
            lng = longitude
        try:
            nearest_grid = pingrid.sel_snap(rr_mrg.precip, lat, lng)
            lat = nearest_grid["Y"].values
            lng = nearest_grid["X"].values
        except KeyError:
            lat = lat
            lng = lng
    return [lat, lng], lat, lng

@SERVER.route(f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>")
def overlay_layers(tz, tx, ty):
    parse_arg = pingrid.parse_arg
    map_choice = parse_arg("map_choice")
    season_choice = parse_arg("season_choice")   

    x_min = pingrid.tile_left(tx, tz)
    x_max = pingrid.tile_left(tx + 1, tz)

    # row numbers increase as latitude decreases
    y_max = pingrid.tile_top_mercator(ty, tz)
    y_min = pingrid.tile_top_mercator(ty + 1, tz)
    precip = rr_mrg.precip
    
    if (
            # When we generalize this to other datasets, remember to
            # account for the possibility that longitudes wrap around,
            # so a < b doesn't always mean that a is west of b.
            x_min > precip['X'].max() or
            x_max < precip['X'].min() or
            y_min > precip['Y'].max() or
            y_max < precip['Y'].min()
    ):
        print("empty")
        return pingrid.image_resp(pingrid.empty_tile())
    precip_tile = precip_tile.sel(
        X=slice(x_min - x_min % RESOLUTION, x_max + RESOLUTION - x_max % RESOLUTION),
        Y=slice(y_min - y_min % RESOLUTION, y_max + RESOLUTION - y_max % RESOLUTION),
    ).compute()
    mymap_min = np.timedelta64(0)
    mymap_min = np.timedelta64(100)

    mycolormap = pingrid.RAINBOW_COLORMAP

    mymap = precip_tile.mean("T")
    mymap.attrs["colormap"] = mycolormap
    mymap = mymap.rename(X="lon", Y="lat")
    mymap.attrs["scale_min"] = mymap_min
    mymap.attrs["scale_max"] = mymap_max
    result = pingrid.tile(mymap, tx, ty, tz, clip_shape)
    print(mymap)
    return result

@APP.callback(
    Output("colorbar", "children"),
    Output("colorbar", "colorscale"),
    Output("colorbar", "max"),
    Output("colorbar", "tickValues"),
    Input("map_choice", "value"),
)
def set_colorbar(
    map_choice,
):
    mymap_max = 60 #taw.max()
    return (
        f"{CONFIG['map_text'][map_choice]['menu_label']} [{CONFIG['map_text'][map_choice]['units']}]",
        pingrid.to_dash_colorscale(pingrid.RAINFALL_COLORMAP),
        mymap_max,
        [i for i in range(0, mymap_max + 1) if i % int(mymap_max/12) == 0],
    )

if __name__ == "__main__":
    APP.run_server(
        debug=GLOBAL_CONFIG["mode"] != "prod",
        processes=GLOBAL_CONFIG["dev_processes"],
        threaded=False,
    )
