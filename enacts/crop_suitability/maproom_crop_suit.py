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
from pingrid import CMAPS
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

PFX = f'{GLOBAL_CONFIG["url_path_prefix"]}{CONFIG["core_path"]}'
TILE_PFX = "/tile"

with psycopg2.connect(**GLOBAL_CONFIG["db"]) as conn:
    s = sql.Composed([sql.SQL(GLOBAL_CONFIG["datasets"]['shapes_adm'][0]['sql'])])
    df = pd.read_sql(s, conn)
    clip_shape = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))[0]

# Reads daily data
rr_mrg = calc.read_zarr_data(Path(
    f'{GLOBAL_CONFIG["datasets"]["daily"]["zarr_path"]}{GLOBAL_CONFIG["datasets"]["daily"]["vars"]["precip"][1]}'
))[CONFIG["layers"]["precip_layer"]["id"]]
tmin_mrg = calc.read_zarr_data(Path(
    f'{GLOBAL_CONFIG["datasets"]["daily"]["zarr_path"]}{GLOBAL_CONFIG["datasets"]["daily"]["vars"]["tmin"][1]}'
))[CONFIG["layers"]["tmin_layer"]["id"]]
tmax_mrg = calc.read_zarr_data(Path(
    f'{GLOBAL_CONFIG["datasets"]["daily"]["zarr_path"]}{GLOBAL_CONFIG["datasets"]["daily"]["vars"]["tmax"][1]}'
))[CONFIG["layers"]["tmax_layer"]["id"]]
# Assumes that grid spacing is regular and cells are square. When we
# generalize this, don't make those assumptions.
RESOLUTION = rr_mrg['X'][1].item() - rr_mrg['X'][0].item()
# The longest possible distance between a point and the center of the
# grid cell containing that point.

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
    Input("submit_coords","n_clicks"),
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
    rainfall_data,
    min_wet_days,
    wet_day_def,
    tmax_data,
    tmin_data,
    lower_wet_threshold,
    upper_wet_threshold,
    max_temp,
    min_temp,
    temp_range,
    target_season,
):
    seasonal_precip = rainfall_data.sel(
        T=rainfall_data['T.season']==target_season
    ).load()
    seasonal_tmax = tmax_data.sel(T=tmax_data['T.season']==target_season).load()
    seasonal_tmin = tmin_data.sel(T=tmin_data['T.season']==target_season).load()
    sum_precip = seasonal_precip.groupby("T.year").sum("T")
    avg_tmax = seasonal_tmax.groupby("T.year").mean("T")
    avg_tmin = seasonal_tmin.groupby("T.year").mean("T")

    #calculate average daily temperature range
    avg_daily_temp_range = (
        seasonal_tmax - seasonal_tmin
    ).groupby("T.year").mean("T")
    
    total_wet_days = xr.where(
        seasonal_precip >= float(wet_day_def),1,0
    ).groupby("T.year").sum("T")
    
    #calculate total precip
    total_precip_range = xr.where(
        np.logical_and(
            sum_precip <= float(upper_wet_threshold), 
            sum_precip >= float(lower_wet_threshold)
        ),1, 0)
    
    tmax = xr.where(avg_tmax <= float(max_temp), 1, 0)
    tmin = xr.where(avg_tmin >= float(min_temp), 1, 0)
    avg_temp_range = xr.where(avg_daily_temp_range <= float(temp_range), 1, 0)
    wet_days = xr.where(total_wet_days >= float(min_wet_days), 1, 0)
    
    crop_suitability = xr.Dataset(
        data_vars = dict(
        ),
        coords = dict(
            X = avg_tmax["X"],
            Y = avg_tmax["Y"],
            year = avg_tmax["year"],
        ), 
    )
    crop_suitability = crop_suitability.assign(
        max_temp = tmax, min_temp = tmin, 
        temp_range = avg_temp_range, precip_range = total_precip_range, 
        wet_days = wet_days)
    crop_suitability['crop_suit'] = (
        crop_suitability['max_temp'] + crop_suitability['min_temp'] + 
        crop_suitability['temp_range'] + crop_suitability['precip_range'] + 
        crop_suitability['wet_days'])

    crop_suitability = crop_suitability.dropna(
        dim="year", how="any"
    ).rename({"year":"T"})
    
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
        if data_choice == "precip_layer":
            data_var = pingrid.sel_snap(rr_mrg, lat1, lng1)
            isnan = np.isnan(data_var).sum()
        elif data_choice == "suitability_layer":
            rr_mrg_sel = pingrid.sel_snap(rr_mrg, lat1, lng1)
            tmax_mrg_sel = pingrid.sel_snap(tmax_mrg, lat1, lng1)
            tmin_mrg_sel = pingrid.sel_snap(tmin_mrg, lat1, lng1)
            data_var = crop_suitability(
                rr_mrg_sel, min_wet_days, wet_day_def, tmax_mrg_sel, tmin_mrg_sel,
                lower_wet_threshold, upper_wet_threshold, maximum_temp,
                minimum_temp, temp_range, target_season
            )
            isnan = np.isnan(data_var["crop_suit"]).sum()
        elif data_choice == "tmax_layer":
            data_var = pingrid.sel_snap(tmax_mrg, lat1, lng1)
            isnan = np.isnan(data_var).sum()
        elif data_choice == "tmin_layer":
            data_var = pingrid.sel_snap(tmin_mrg, lat1, lng1)
            isnan = np.isnan(data_var).sum()
        if isnan > 0:
            error_fig = pingrid.error_fig(error_msg="Data missing at this location")
            return error_fig
    except KeyError:
        error_fig = pingrid.error_fig(error_msg="Grid box out of data domain")
        return error_fig

    if data_choice == "suitability_layer":
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
                'range' : [CONFIG['layers'][data_choice]['map_min'],CONFIG['layers'][data_choice]['map_max']],
                'tickvals' : [*range(CONFIG['layers'][data_choice]['map_min'], CONFIG['layers'][data_choice]['map_max']+1)],
                'tickformat':',d'
            },
            xaxis_title = "years",
            yaxis_title = "Suitability index",
            title = f"{CONFIG['layers'][data_choice]['menu_label']} for season {season_str}, coordinates: [{lat1}N, {lng1}E]"
        ) 
    else:
        seasonal_var = data_var.sel(T=data_var['T.season']==target_season)
        if data_choice == "precip_layer":
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
            yaxis_title = f"{CONFIG['layers'][data_choice]['id']} ({CONFIG['layers'][data_choice]['units']})",
            title = f"{CONFIG['layers'][data_choice]['menu_label']} for season {season_str}, coordinates: [{lat1}N, {lng1}E]"
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
    Input("target_year","value"),
    Input("target_season","value"),
)
def write_map_title(target_year,target_season):
    season_str = select_season(target_season)
    map_title = ("Climate suitability for " + season_str + " season in " + str(target_year))

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
    mymap_min = CONFIG["layers"][data_choice]["map_min"]
    mymap_max = CONFIG["layers"][data_choice]["map_max"]
    rr_mrg_year = rr_mrg.sel(T=rr_mrg['T.year']==target_year)
    rr_mrg_season = rr_mrg_year.sel(T=rr_mrg_year["T.season"] == target_season)
    tmin_mrg_year = tmin_mrg.sel(T=tmin_mrg['T.year']==target_year)
    tmin_mrg_season = tmin_mrg_year.sel(T=tmin_mrg_year["T.season"] == target_season)
    tmax_mrg_year = tmax_mrg.sel(T=tmax_mrg['T.year']==target_year)
    tmax_mrg_season = tmax_mrg_year.sel(T=tmax_mrg_year["T.season"] == target_season)

    if data_choice == "suitability_layer":
        crop_suit_vals = crop_suitability(
            rr_mrg_year, min_wet_days, wet_day_def, tmax_mrg_year, tmin_mrg_year, 
            lower_wet_threshold, upper_wet_threshold, maximum_temp,
            minimum_temp, temp_range, target_season 
            ) 
        data_tile = crop_suit_vals["crop_suit"]
    else:
        data_var = CONFIG["layers"][data_choice]["id"]
        if data_choice == "precip_layer":
            data_tile = rr_mrg_season
        if data_choice == "tmin_layer":
            data_tile = tmin_mrg_season
        if data_choice == "tmax_layer":
            data_tile = tmax_mrg_season
    if (
            # When we generalize this to other datasets, remember to
            # account for the possibility that longitudes wrap around,
            # so a < b doesn't always mean that a is west of b.
            x_min > data_tile['X'].max() or
            x_max < data_tile['X'].min() or
            y_min > data_tile['Y'].max() or
            y_max < data_tile['Y'].min()
    ):
        return pingrid.image_resp(pingrid.empty_tile())

    data_tile = data_tile.sel(
        X=slice(x_min - x_min % RESOLUTION, x_max + RESOLUTION - x_max % RESOLUTION),
        Y=slice(y_min - y_min % RESOLUTION, y_max + RESOLUTION - y_max % RESOLUTION),
    ).compute()

    mycolormap = CMAPS["rainbow"]

    if data_choice == "suitability_layer":
        mymap = data_tile
    elif data_choice == "precip_layer":
        mymap = data_tile.sum("T")
    else:
        mymap = data_tile.mean("T")

    mymap = np.squeeze(mymap)
    mymap.attrs["colormap"] = mycolormap
    mymap = mymap.rename(X="lon", Y="lat")
    mymap.attrs["scale_min"] = mymap_min
    mymap.attrs["scale_max"] = mymap_max
    result = pingrid.tile(mymap.astype('float64'), tx, ty, tz, clip_shape)

    return result


@APP.callback(
    Output("colorbar", "children"),
    Output("colorbar", "colorscale"),
    Output("colorbar", "max"),
    Output("colorbar", "tickValues"),
    Input("data_choice", "value"),
)
def set_colorbar(
    data_choice,
):
    mymap_max = CONFIG["layers"][data_choice]["map_max"]
    mymap_min = CONFIG["layers"][data_choice]["map_min"]
    if data_choice == "suitability_layer":
        tick_freq = 1
    else:
        tick_freq = 5
    return (
        f"{CONFIG['layers'][data_choice]['menu_label']} [{CONFIG['layers'][data_choice]['units']}]",
        CMAPS["rainbow"].to_dash_leaflet(),
        mymap_max,
        [i for i in range(mymap_min, mymap_max + 1) if i % tick_freq == 0],
    )

