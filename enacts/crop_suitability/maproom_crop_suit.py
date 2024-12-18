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
from pingrid import CMAPS, BROWN, YELLOW, ORANGE, PALEGREEN, GREEN, DARKGREEN
from . import layout_crop_suit
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
import xarray as xr

from globals_ import FLASK, GLOBAL_CONFIG

CONFIG = GLOBAL_CONFIG["maprooms"]["crop_suitability"]

PFX = f'{GLOBAL_CONFIG["url_path_prefix"]}/{CONFIG["core_path"]}'
TILE_PFX = f"{PFX}/tile"

with psycopg2.connect(**GLOBAL_CONFIG["db"]) as conn:
    s = sql.Composed([sql.SQL(GLOBAL_CONFIG["datasets"]['shapes_adm'][0]['sql'])])
    df = pd.read_sql(s, conn)
    clip_shape = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))[0]

# Reads daily data
zarr_path_rr = GLOBAL_CONFIG["datasets"]["daily"]["vars"]["precip"][1]
if zarr_path_rr is None:
    zarr_path_rr = GLOBAL_CONFIG["datasets"]["daily"]["vars"]["precip"][0]
rr_mrg = calc.read_zarr_data(Path(
    f'{GLOBAL_CONFIG["datasets"]["daily"]["zarr_path"]}{zarr_path_rr}'
))[GLOBAL_CONFIG["datasets"]["daily"]["vars"]["precip"][2]]
zarr_path_tmin = GLOBAL_CONFIG["datasets"]["daily"]["vars"]["tmin"][1]
if zarr_path_tmin is None:
    zarr_path_tmin = GLOBAL_CONFIG["datasets"]["daily"]["vars"]["tmin"][0]
tmin_mrg = calc.read_zarr_data(Path(
    f'{GLOBAL_CONFIG["datasets"]["daily"]["zarr_path"]}{zarr_path_tmin}'
))[GLOBAL_CONFIG["datasets"]["daily"]["vars"]["tmin"][2]]
zarr_path_tmax = GLOBAL_CONFIG["datasets"]["daily"]["vars"]["tmax"][1]
if zarr_path_tmax is None:
    zarr_path_tmax = GLOBAL_CONFIG["datasets"]["daily"]["vars"]["tmax"][0]
tmax_mrg = calc.read_zarr_data(Path(
    f'{GLOBAL_CONFIG["datasets"]["daily"]["zarr_path"]}{zarr_path_tmax}'
))[GLOBAL_CONFIG["datasets"]["daily"]["vars"]["tmax"][2]]
# Assumes that grid spacing is regular and cells are square. When we
# generalize this, don't make those assumptions.
RESOLUTION = rr_mrg['X'][1].item() - rr_mrg['X'][0].item()
# The longest possible distance between a point and the center of the
# grid cell containing that point.

CROP_SUIT_COLORMAP = pingrid.ColorScale(
    "crop_suit",
    [BROWN, BROWN, ORANGE, ORANGE, YELLOW, YELLOW,
        PALEGREEN, PALEGREEN, GREEN, GREEN, DARKGREEN, DARKGREEN],
    [0, 0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5, 4.5, 5],
)

APP = dash.Dash(
    __name__,
    server=FLASK,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
    ],
    url_base_pathname=f"{PFX}/",
    meta_tags=[
        {"name": "description", "content": "Onset Maproom"},
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
                "fill": False,
                "color": adm_color,
                "weight": adm_weight,
            },
        ),
        name=adm_name,
        checked=is_checked,
    )

@APP.callback(
    Output("layers_control", "children"),
    Input("submit_params", "n_clicks"),
    Input("data_choice", "value"),
    Input("target_season", "value"),
    Input("target_year", "value"),
    State("min_wet_days","value"),
    State("wet_day_def","value"),
    State("lower_wet_threshold","value"),
    State("upper_wet_threshold","value"),
    State("maximum_temp","value"),
    State("minimum_temp","value"),
    State("temp_range","value"),
)
def make_map(
        n_clicks,
        data_choice,
        target_season,
        target_year,
        min_wet_days,
        wet_day_def,
        lower_wet_threshold,
        upper_wet_threshold,
        maximum_temp,
        minimum_temp,
        temp_range,
):
    qstr = urllib.parse.urlencode({
        "data_choice": data_choice,
        "target_season": target_season,
        "target_year": target_year,
        "min_wet_days": min_wet_days,
        "wet_day_def": wet_day_def,
        "lower_wet_threshold": lower_wet_threshold,
        "upper_wet_threshold": upper_wet_threshold,
        "maximum_temp": maximum_temp,
        "minimum_temp": minimum_temp,
        "temp_range": temp_range,
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
            len(GLOBAL_CONFIG["datasets"]["shapes_adm"])-i,
            is_checked=adm["is_checked"]
        )
        for i, adm in enumerate(GLOBAL_CONFIG["datasets"]["shapes_adm"])
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
        lat = rr_mrg["Y"][int(rr_mrg["Y"].size/2)].values
        lng = rr_mrg["X"][int(rr_mrg["X"].size/2)].values
    else:
        if dash.ctx.triggered_id == "map":
            lat = click_lat_lng[0]
            lng = click_lat_lng[1]
        else:
            lat = latitude
            lng = longitude
        try:
            nearest_grid = pingrid.sel_snap(rr_mrg, lat, lng)
            lat = nearest_grid["Y"].values
            lng = nearest_grid["X"].values
        except KeyError:
            lat = lat
            lng = lng
    return [lat, lng], lat, lng


def crop_suitability(
    rainfall,
    min_wet_days,
    wet_day_def,
    tmax,
    tmin,
    lower_wet_threshold,
    upper_wet_threshold,
    max_temp,
    min_temp,
    temp_range,
    target_season,
):
    seasonal_precip = rainfall.sel(T=rainfall['T.season']==target_season)
    seasonal_tmax = tmax.sel(T=tmax['T.season']==target_season)
    seasonal_tmin = tmin.sel(T=tmin['T.season']==target_season)

    seasonal_avg_tmax_suitability = (
        seasonal_tmax.groupby("T.year").mean() <= max_temp
    )
    seasonal_avg_tmin_suitability = (
        seasonal_tmin.groupby("T.year").mean() >= min_temp
    )

    seasonal_avg_temp_amplitude_suitability = (
        (seasonal_tmax - seasonal_tmin).groupby("T.year").mean() <= temp_range
    )
    
    seasonal_wet_days_suitability = (
        (seasonal_precip >= wet_day_def).groupby("T.year").sum() >= min_wet_days
    )

    seasonal_total_precip_suitability = (
        (seasonal_precip.groupby("T.year").sum() <= upper_wet_threshold) &
        (seasonal_precip.groupby("T.year").sum() >= lower_wet_threshold)
    )

    crop_suit = (
        seasonal_avg_tmax_suitability.astype(int) +
        seasonal_avg_tmin_suitability.astype(int) +
        seasonal_avg_temp_amplitude_suitability.astype(int) +
        seasonal_total_precip_suitability.astype(int) +
        seasonal_wet_days_suitability.astype(int)
    )
    
    crop_suitability = xr.Dataset(
        data_vars = dict(
            max_temp = seasonal_avg_tmax_suitability,
            min_temp = seasonal_avg_tmin_suitability,
            temp_range = seasonal_avg_temp_amplitude_suitability,
            precip_range = seasonal_total_precip_suitability,
            wet_days = seasonal_wet_days_suitability,
            crop_suit = crop_suit,
        ),
        coords = dict(
            X = seasonal_avg_tmax_suitability["X"],
            Y = seasonal_avg_tmax_suitability["Y"],
            year = seasonal_avg_tmax_suitability["year"],
        ), 
    ).dropna(dim="year", how="any").rename({"year":"T"})

    return crop_suitability

@APP.callback(
    Output("timeseries_graph","figure"),
    Input("loc_marker", "position"),
    Input("data_choice","value"),
    Input("submit_params","n_clicks"),
    Input("target_season","value"),
    State("lower_wet_threshold","value"),
    State("upper_wet_threshold","value"),
    State("minimum_temp","value"),
    State("maximum_temp","value"),
    State("temp_range","value"),
    State("min_wet_days","value"),
    State("wet_day_def","value"),
)
def timeseries_plot(
    loc_marker,
    data_choice,
    n_clicks,
    target_season,
    lower_wet_threshold,
    upper_wet_threshold,
    minimum_temp,
    maximum_temp,
    temp_range,
    min_wet_days,
    wet_day_def,
):
    lat1 = loc_marker[0]
    lng1 = loc_marker[1]
    season_str = select_season(target_season)
    try:
        if data_choice == "precip":
            data_var = pingrid.sel_snap(rr_mrg, lat1, lng1)
            isnan = np.isnan(data_var).sum()
        elif data_choice == "suitability":
            rr_mrg_sel = pingrid.sel_snap(rr_mrg, lat1, lng1)
            tmax_mrg_sel = pingrid.sel_snap(tmax_mrg, lat1, lng1)
            tmin_mrg_sel = pingrid.sel_snap(tmin_mrg, lat1, lng1)
            data_var = crop_suitability(
                rr_mrg_sel, int(min_wet_days), float(wet_day_def),
                tmax_mrg_sel, tmin_mrg_sel,
                float(lower_wet_threshold), float(upper_wet_threshold),
                float(maximum_temp), float(minimum_temp), float(temp_range),
                target_season,
            )
            isnan = np.isnan(data_var["crop_suit"]).sum()
        elif data_choice == "tmax":
            data_var = pingrid.sel_snap(tmax_mrg, lat1, lng1)
            isnan = np.isnan(data_var).sum()
        elif data_choice == "tmin":
            data_var = pingrid.sel_snap(tmin_mrg, lat1, lng1)
            isnan = np.isnan(data_var).sum()
        if isnan > 0:
            error_fig = pingrid.error_fig(error_msg="Data missing at this location")
            return error_fig
    except KeyError:
        error_fig = pingrid.error_fig(error_msg="Grid box out of data domain")
        return error_fig

    if data_choice == "suitability":
        seasonal_suit = data_var
        timeseries_plot = pgo.Figure()
        timeseries_plot.add_trace(
            pgo.Bar(
                x = seasonal_suit["T"].values,
                y = seasonal_suit["crop_suit"].where(
                    # 0 is a both legitimate start for bars and data value
                    # but in that case 0 won't draw a bar, and the is nothing to hover
                    # this giving a dummy small height to draw a bar to hover
                    lambda x: x > 0, other=0.1
                ).values,
            )
        )
        timeseries_plot.update_layout(
            yaxis={
                'range' : [0, 5],
                'tickvals' : [*range(0, 5+1)],
                'tickformat':',d'
            },
            xaxis_title = "years",
            yaxis_title = "Suitability index",
            title = f"{CONFIG['map_text'][data_choice]['menu_label']} for {season_str} at [{lat1}N, {lng1}E]"
        ) 
    else:
        seasonal_var = data_var.sel(T=data_var['T.season']==target_season)
        if data_choice == "precip":
            seasonal_mean = seasonal_var.groupby("T.year").sum("T").rename({"year":"T"})
        else:
            seasonal_mean = seasonal_var.groupby("T.year").mean("T").rename({"year":"T"})
        
        timeseries_plot = pgo.Figure()
        timeseries_plot.add_trace(
            pgo.Scatter(
                x = seasonal_mean["T"].values,
                y = seasonal_mean.values,
                line=pgo.scatter.Line(color="blue"),
            )
        )
        timeseries_plot.update_traces(mode="lines", connectgaps=False)
        timeseries_plot.update_layout(
            xaxis_title = "years",
            yaxis_title = f"{CONFIG['map_text'][data_choice]['id']} ({CONFIG['map_text'][data_choice]['units']})",
            title = f"{CONFIG['map_text'][data_choice]['menu_label']} for {season_str} at [{lat1}N, {lng1}E]"
        )

    return timeseries_plot

def select_season(target_season):
    if target_season == 'MAM':
        season_str = 'Mar-May'
    if target_season == 'JJA':
        season_str = 'Jun-Aug'
    if target_season == 'SON':
        season_str = 'Sep-Nov'
    if target_season == 'DJF':
        season_str = 'Dec-Feb'    
    return season_str 

@APP.callback(
    Output("map_title","children"),
    Input("data_choice","value"),
    Input("target_year","value"),
    Input("target_season","value"),
)
def write_map_title(data_choice, target_year, target_season):
    season_str = select_season(target_season)
    map_title = f"{CONFIG['map_text'][data_choice]['menu_label']} for {season_str} in {str(target_year)}"

    return map_title

@FLASK.route(f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>")
def cropSuit_layers(tz, tx, ty):
    parse_arg = pingrid.parse_arg
    data_choice = parse_arg("data_choice")
    target_season = parse_arg("target_season")
    target_year = parse_arg("target_year", float)  
    data_choice = parse_arg("data_choice")
    min_wet_days = parse_arg("min_wet_days", int)
    wet_day_def = parse_arg("wet_day_def", float)
    lower_wet_threshold = parse_arg("lower_wet_threshold", int)
    upper_wet_threshold = parse_arg("upper_wet_threshold", int)
    maximum_temp = parse_arg("maximum_temp", float)
    minimum_temp = parse_arg("minimum_temp", float)
    temp_range = parse_arg("temp_range", float) 

    x_min = pingrid.tile_left(tx, tz)
    x_max = pingrid.tile_left(tx + 1, tz)

    # row numbers increase as latitude decreases
    y_max = pingrid.tile_top_mercator(ty, tz)
    y_min = pingrid.tile_top_mercator(ty + 1, tz)

    if (
            # When we generalize this to other datasets, remember to
            # account for the possibility that longitudes wrap around,
            # so a < b doesn't always mean that a is west of b.
            x_min > rr_mrg['X'].max() or
            x_max < rr_mrg['X'].min() or
            y_min > rr_mrg['Y'].max() or
            y_max < rr_mrg['Y'].min()
    ):
        return pingrid.image_resp(pingrid.empty_tile())

    rr_mrg_year = rr_mrg.sel(T=rr_mrg['T.year']==target_year)
    tmin_mrg_year = tmin_mrg.sel(T=tmin_mrg['T.year']==target_year)
    tmax_mrg_year = tmax_mrg.sel(T=tmax_mrg['T.year']==target_year)

    rr_mrg_year_tile = rr_mrg_year.sel(
        X=slice(x_min - x_min % RESOLUTION, x_max + RESOLUTION - x_max % RESOLUTION),
        Y=slice(y_min - y_min % RESOLUTION, y_max + RESOLUTION - y_max % RESOLUTION),
    )

    tmin_mrg_year_tile = tmin_mrg_year.sel(
        X=slice(x_min - x_min % RESOLUTION, x_max + RESOLUTION - x_max % RESOLUTION),
        Y=slice(y_min - y_min % RESOLUTION, y_max + RESOLUTION - y_max % RESOLUTION),
    )

    tmax_mrg_year_tile = tmax_mrg_year.sel(
        X=slice(x_min - x_min % RESOLUTION, x_max + RESOLUTION - x_max % RESOLUTION),
        Y=slice(y_min - y_min % RESOLUTION, y_max + RESOLUTION - y_max % RESOLUTION),
    )

    rr_mrg_season = rr_mrg_year_tile.sel(T=rr_mrg_year_tile["T.season"] == target_season)
    tmin_mrg_season = tmin_mrg_year_tile.sel(T=tmin_mrg_year_tile["T.season"] == target_season)
    tmax_mrg_season = tmax_mrg_year_tile.sel(T=tmax_mrg_year_tile["T.season"] == target_season)

    if data_choice == "suitability":
        map_min = 0
        map_max = 5
        crop_suit_vals = crop_suitability(
            rr_mrg_year_tile, int(min_wet_days), float(wet_day_def),
            tmax_mrg_year_tile, tmin_mrg_year_tile,
            float(lower_wet_threshold), float(upper_wet_threshold),
            float(maximum_temp), float(minimum_temp), float(temp_range),
            target_season,
        ) 
        data_tile = crop_suit_vals["crop_suit"]
    else:
        map_min = CONFIG["map_text"][data_choice]["map_min"]
        map_max = CONFIG["map_text"][data_choice]["map_max"]
        if data_choice == "precip":
            data_tile = rr_mrg_season
        if data_choice == "tmin":
            data_tile = tmin_mrg_season
        if data_choice == "tmax":
            data_tile = tmax_mrg_season

    if data_choice == "suitability":
        map = data_tile
        colormap = CROP_SUIT_COLORMAP
    elif data_choice == "precip":
        map = data_tile.sum("T")
        colormap = CMAPS["precip"]
    else:
        map = data_tile.mean("T")
        colormap = CMAPS["temp"]

    map = np.squeeze(map)
    map.attrs["colormap"] = colormap
    map = map.rename(X="lon", Y="lat")
    map.attrs["scale_min"] = map_min
    map.attrs["scale_max"] = map_max
    result = pingrid.tile(map.astype('float64'), tx, ty, tz, clip_shape)

    return result


@APP.callback(
    Output("colorbar", "children"),
    Output("colorbar", "colorscale"),
    Output("colorbar", "min"),
    Output("colorbar", "max"),
    Output("colorbar", "tickValues"),
    Input("data_choice", "value"),
)
def set_colorbar(
    data_choice,
):
    if data_choice == "suitability":
        colormap = CROP_SUIT_COLORMAP
        map_min = 0
        map_max = 5
        tick_freq = 1
    else:
        map_min = CONFIG["map_text"][data_choice]["map_min"]
        map_max = CONFIG["map_text"][data_choice]["map_max"]
        if data_choice == "precip":
            colormap = CMAPS["precip"]
            tick_freq = 50
        else:
            colormap = CMAPS["temp"]
            tick_freq = 4 
    return (
        f"{CONFIG['map_text'][data_choice]['menu_label']} [{CONFIG['map_text'][data_choice]['units']}]",
        colormap.to_dash_leaflet(),
        map_min,
        map_max,
        [i for i in range(map_min, map_max + 1) if i % tick_freq == 0],
    )

