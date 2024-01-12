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
from . import layout
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

from globals_ import FLASK, GLOBAL_CONFIG

CONFIG = GLOBAL_CONFIG["maprooms"]["onset"]

PFX = f'{GLOBAL_CONFIG["url_path_prefix"]}/{CONFIG["core_path"]}'
TILE_PFX = f"{PFX}/tile"

with psycopg2.connect(**GLOBAL_CONFIG["db"]) as conn:
    s = sql.Composed([sql.SQL(GLOBAL_CONFIG['datasets']['shapes_adm'][0]['sql'])])
    df = pd.read_sql(s, conn)
    clip_shape = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))[0]

# Reads daily data

DATA_PATH = GLOBAL_CONFIG['datasets']['daily']['vars']['precip'][1]
if DATA_PATH is None:
    DATA_PATH = GLOBAL_CONFIG['datasets']['daily']['vars']['precip'][0]
DR_PATH = f"{GLOBAL_CONFIG['datasets']['daily']['zarr_path']}{DATA_PATH}"
RR_MRG_ZARR = Path(DR_PATH)
rr_mrg = calc.read_zarr_data(RR_MRG_ZARR)

if (rr_mrg["T"].diff("T") <= np.timedelta64(0)).any():
    raise Exception(
        "Input time dimension must be strictly increasing"
    )

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
APP.title = CONFIG["title"]

APP.layout = layout.app_layout()


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
    Input("search_start_day", "value"),
    Input("search_start_month", "value"),
    Input("search_days", "value"),
    Input("wet_threshold", "value"),
    Input("running_days", "value"),
    Input("running_total", "value"),
    Input("min_rainy_days", "value"),
    Input("dry_days", "value"),
    Input("dry_spell", "value"),
    Input("cess_start_day", "value"),
    Input("cess_start_month", "value"),
    Input("cess_search_days", "value"),
    Input("cess_soil_moisture", "value"),
    Input("cess_dry_spell", "value"),
    Input("prob_exc_thresh_onset", "value"),
    Input("prob_exc_thresh_length", "value"),
    Input("prob_exc_thresh_tot", "value"),
)
def make_map(
        map_choice,
        search_start_day,
        search_start_month,
        search_days,
        wet_thresh,
        wet_spell_length,
        wet_spell_thresh,
        min_wet_days,
        dry_spell_length,
        dry_spell_search,
        cess_start_day,
        cess_start_month,
        cess_search_days, 
        cess_soil_moisture,
        cess_dry_spell,
        prob_exc_thresh_onset,
        prob_exc_thresh_length,
        prob_exc_thresh_tot,
):
    qstr = urllib.parse.urlencode({
        "map_choice": map_choice,
        "search_start_day": search_start_day,
        "search_start_month": search_start_month,
        "search_days": search_days,
        "wet_thresh": wet_thresh,
        "wet_spell_length": wet_spell_length,
        "wet_spell_thresh": wet_spell_thresh,
        "min_wet_days": min_wet_days,
        "dry_spell_length": dry_spell_length,
        "dry_spell_search": dry_spell_search,
        "cess_start_day": cess_start_day,
        "cess_start_month": cess_start_month,
        "cess_search_days": cess_search_days,
        "cess_soil_moisture": cess_soil_moisture,
        "cess_dry_spell": cess_dry_spell,
        "prob_exc_thresh_onset": prob_exc_thresh_onset,
        "prob_exc_thresh_length": prob_exc_thresh_length,
        "prob_exc_thresh_tot": prob_exc_thresh_tot,
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
            name="Onset",
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
    Output("pet_input_wrap_onset", "style"),
    Output("pet_input_wrap_length", "style"),
    Output("pet_input_wrap_tot", "style"),
    Input("map_choice", "value"),
)
def display_pet_control(map_choice):

    pet_input_wrap_onset={"display": "none"}
    pet_input_wrap_length={"display": "none"}
    pet_input_wrap_tot={"display": "none"}
    if map_choice == "pe":
        pet_input_wrap_onset={"display": "flex"}
    elif map_choice == "length_pe":
        pet_input_wrap_length={"display": "flex"}
    elif map_choice == "total_pe":
        pet_input_wrap_tot={"display": "flex"}
    return pet_input_wrap_onset, pet_input_wrap_length, pet_input_wrap_tot


@APP.callback(
    Output("pet_units1", "children"),
    Input("search_start_day", "value"),
    Input("search_start_month", "value"),
)
def write_pet_units(search_start_day, search_start_month):

    return "days after " + search_start_month + " " + search_start_day


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
    Output("map_title", "children"),
    Input("search_start_day", "value"),
    Input("search_start_month", "value"),
    Input("map_choice", "value"),
    Input("prob_exc_thresh_onset", "value"),
    Input("prob_exc_thresh_length", "value"),
    Input("prob_exc_thresh_tot", "value"),
)
def write_map_title(search_start_day, search_start_month, map_choice, pet_onset, pet_length, pet_tot):

    if map_choice == "monit":
        search_start_month1 = calc.strftimeb2int(search_start_month)
        first_day = calc.sel_day_and_month(
            rr_mrg.precip["T"][-366:-1], int(search_start_day), search_start_month1
        )[0].dt.strftime("%Y-%m-%d").values
        last_day = rr_mrg.precip["T"][-1].dt.strftime("%Y-%m-%d").values
        map_title = (
            "Onset date found between " + first_day + " and " + last_day
            + " in days since " + first_day
        )
    if map_choice == "mean":
        map_title = (
            "Climatological Onset date in days since " 
            + search_start_month + " " + search_start_day
        )
    if map_choice == "stddev":
        map_title = (
            "Climatological Onset date standard deviation in days since "
            + search_start_month + " " + search_start_day
        )
    if map_choice == "pe":
        map_title = (
            "Climatological probability that Onset date is " + pet_onset  + " days past "
            + search_start_month + " " + search_start_day
        )
    if map_choice == "length_mean":
        map_title = (
            "Climatological Length of season in days"
        )
    if map_choice == "length_stddev":
        map_title = (
            "Climatological Length of season standard deviation in days"
        )
    if map_choice == "length_pe":
        map_title = (
            "Climatological probability that season is shorter than "
            + pet_length  + " days"
        )
    if map_choice == "total_mean":
        map_title = (
            "Climatological Total seasonal precipitation in mm"
        )
    if map_choice == "total_stddev":
        map_title = (
            "Climatological Total seasoanl precipitation standard deviation in mm"
        )
    if map_choice == "total_pe":
        map_title = (
            "Climatological probability that it rains less than "
            + pet_tot  + " mm in season"
        )
    return map_title


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


@APP.callback(
    Output("onset_date_plot", "figure"),
    Output("onset_prob_exc", "figure"),
    Output("germination_sentence", "children"),
    Input("loc_marker", "position"),
    Input("search_start_day", "value"),
    Input("search_start_month", "value"),
    Input("search_days", "value"),
    Input("wet_threshold", "value"),
    Input("running_days", "value"),
    Input("running_total", "value"),
    Input("min_rainy_days", "value"),
    Input("dry_days", "value"),
    Input("dry_spell", "value"),
)
def onset_plots(
    marker_pos,
    search_start_day,
    search_start_month,
    search_days,
    wet_threshold,
    running_days,
    running_total,
    min_rainy_days,
    dry_days,
    dry_spell,
):
    lat = marker_pos[0]
    lng = marker_pos[1]
    try:
        precip = pingrid.sel_snap(rr_mrg.precip, lat, lng)
        isnan = np.isnan(precip).any()
        if isnan:
            error_fig = pingrid.error_fig(error_msg="Data missing at this location")
            germ_sentence = ""
            return error_fig, error_fig, germ_sentence
    except KeyError:
        error_fig = pingrid.error_fig(error_msg="Grid box out of data domain")
        germ_sentence = ""
        return error_fig, error_fig, germ_sentence
    precip.load()
    try:
        onset_delta = calc.seasonal_onset_date(
            precip,
            int(search_start_day),
            calc.strftimeb2int(search_start_month),
            int(search_days),
            int(wet_threshold),
            int(running_days),
            int(running_total),
            int(min_rainy_days),
            int(dry_days),
            int(dry_spell),
            time_dim="T",
        )
        isnan = np.isnan(onset_delta["onset_delta"]).all()
        if isnan:
            error_fig = pingrid.error_fig(error_msg="No onset dates were found")
            germ_sentence = ""
            return error_fig, error_fig, germ_sentence
    except TypeError:
        error_fig = pingrid.error_fig(
            error_msg="Please ensure all input boxes are filled for the calculation to run."
        )
        germ_sentence = ""
        return (
            error_fig,
            error_fig,
            germ_sentence
        )  # dash.no_update to leave the plat as-is and not show no data display
    onset_date_graph = pgo.Figure()
    onset_date_graph.add_trace(
        pgo.Bar(
            x=onset_delta["T"].dt.year.values,
            y=onset_delta["onset_delta"].dt.days.where(
                # 0 is a both legitimate start for bars and data value
                # but in that case 0 won't draw a bar, and the is nothing to hover
                # this giving a dummy small height to draw a bar to hover
                 lambda x: x > 0, other=0.1
            ).values,
            customdata=(onset_delta["T"] + onset_delta["onset_delta"]).dt.strftime("%-d %B %Y"),
            hovertemplate="%{customdata}",
            name="",
        )
    )
    onset_date_graph.update_layout(
        xaxis_title="Year",
        yaxis_title=f"Onset Date in days since {search_start_month} {int(search_start_day)}",
        title=f"Onset dates found after {search_start_month} {int(search_start_day)}, at ({round_latLng(lat)}N,{round_latLng(lng)}E)",
    )
    quantiles = np.arange(1, onset_delta["T"].size + 1) / (onset_delta["T"].size + 1)
    onset_quant = onset_delta["onset_delta"].dt.days.quantile(quantiles, dim="T")
    cdf_graph = pgo.Figure()
    cdf_graph.add_trace(
        pgo.Scatter(
            x=onset_quant.values,
            y=(1 - quantiles),
            customdata=(datetime.datetime(
                2000,
                calc.strftimeb2int(search_start_month),
                int(search_start_day)
            ) + pd.to_timedelta(onset_quant, "D")).strftime("%-d %B"),
            hovertemplate="%{y:.0%} chance of exceeding %{customdata}",
            name="",
            line=pgo.scatter.Line(color="blue"),
        )
    )
    cdf_graph.update_traces(mode="lines", connectgaps=False)
    cdf_graph.update_layout(
        xaxis_title=f"Onset Date in days since {search_start_month} {int(search_start_day)}",
        yaxis_title="Probability of exceeding",
    )
    precip = precip.isel({"T": slice(-366, None)})
    search_start_dm = calc.sel_day_and_month(precip["T"], int(search_start_day), calc.strftimeb2int(search_start_month))
    precip = precip.sel({"T": slice(search_start_dm.values[0], None)})
    germ_rains_date = calc.onset_date(
        precip,
        int(wet_threshold),
        int(running_days),
        int(running_total),
        int(min_rainy_days),
        int(dry_days),
        0
    )
    germ_rains_date = germ_rains_date + germ_rains_date["T"]
    if pd.isnull(germ_rains_date):
        germ_sentence = (
            "Germinating rains have not yet occured as of "
            + precip["T"][-1].dt.strftime("%B %d, %Y").values
        )
    else:
        germ_sentence = (
            "Germinating rains have occured on "
            + germ_rains_date.dt.strftime("%B %d, %Y").values
        )
    #return onsetDate_graph, probExceed_onset, germ_sentence
    return onset_date_graph, cdf_graph, germ_sentence


@APP.callback(
    Output("cess_date_plot", "figure"),
    Output("cess_prob_exc", "figure"),
    Output("cess_tab","tab_style"),
    Input("loc_marker", "position"),
    Input("cess_start_day", "value"),
    Input("cess_start_month", "value"),
    Input("cess_search_days", "value"),
    Input("cess_soil_moisture", "value"),
    Input("cess_dry_spell", "value"),
)
def cess_plots(
    marker_pos,
    cess_start_day,
    cess_start_month,
    cess_search_days,
    cess_soil_moisture,
    cess_dry_spell,
):
    if not CONFIG["ison_cess_date_hist"]:
        tab_style = {"display": "none"}
        return {}, {}, tab_style
    else:
        tab_style = {}
        lat = marker_pos[0]
        lng = marker_pos[1]
        try:
            precip = pingrid.sel_snap(rr_mrg.precip, lat, lng)
            isnan = np.isnan(precip).any()
            if isnan:
                error_fig = pingrid.error_fig(error_msg="Data missing at this location")
                return error_fig, error_fig, tab_style
        except KeyError:
            error_fig = pingrid.error_fig(error_msg="Grid box out of data domain")
            return error_fig, error_fig, tab_style
        precip.load()
        try:
            cess_delta = calc.seasonal_cess_date_from_rain(
                precip,
                int(cess_start_day),
                calc.strftimeb2int(cess_start_month),
                int(cess_search_days),
                int(cess_soil_moisture),
                int(cess_dry_spell),
                5,
                60,
                60./3.,
                time_dim="T",
            )
            isnan = np.isnan(cess_delta["cess_delta"]).all()
            if isnan:
                error_fig = pingrid.error_fig(error_msg="No cessation dates were found")
                return error_fig, error_fig, tab_style
        except TypeError:
            error_fig = pingrid.error_fig(error_msg="Please ensure all input boxes are filled for the calculation to run.")
            return (
                error_fig, error_fig, tab_style,
            )  # dash.no_update to leave the plat as-is and not show no data display
        cess_date_graph = pgo.Figure()
        cess_date_graph.add_trace(
            pgo.Bar(
                x=cess_delta["T"].dt.year.values,
                y=cess_delta["cess_delta"].squeeze().dt.days.where(
                    # 0 is a both legitimate start for bars and data value
                    # but in that case 0 won't draw a bar, and the is nothing to hover
                    # this giving a dummy small height to draw a bar to hover
                    lambda x: x > 0, other=0.1
                ).values,
                customdata=(cess_delta["T"] + cess_delta["cess_delta"]).dt.strftime("%-d %B %Y"),
                hovertemplate="%{customdata}",
                name="",
            )
        )
        cess_date_graph.update_layout(
            xaxis_title="Year",
            yaxis_title=f"Cessation Date in days since {cess_start_month} {int(cess_start_day)}",
            title=f"Cessation dates found after {cess_start_month} {int(cess_start_day)}, at ({round_latLng(lat)}N,{round_latLng(lng)}E)",
        )
        quantiles = np.arange(1, cess_delta["T"].size + 1) / (cess_delta["T"].size + 1)
        cess_quant = cess_delta["cess_delta"].dt.days.quantile(quantiles, dim="T").squeeze()
        cdf_graph = pgo.Figure()
        cdf_graph.add_trace(
            pgo.Scatter(
                x=cess_quant.values,
                y=(1 - quantiles),
                customdata=(datetime.datetime(
                    2000,
                    calc.strftimeb2int(cess_start_month),
                    int(cess_start_day)
                ) + pd.to_timedelta(cess_quant, "D")).strftime("%-d %B"),
                hovertemplate="%{y:.0%} chance of exceeding %{customdata}",
                name="",
                line=pgo.scatter.Line(color="blue"),
            )
        )
        cdf_graph.update_traces(mode="lines", connectgaps=False)
        cdf_graph.update_layout(
            xaxis_title=f"Cessation Date in days since {cess_start_month} {int(cess_start_day)}",
            yaxis_title="Probability of exceeding",
        )
        return cess_date_graph, cdf_graph, tab_style


@APP.callback(
    Output("length_plot", "figure"),
    Output("length_prob_exc", "figure"),
    Output("length_tab","tab_style"),
    Input("loc_marker", "position"),
    Input("search_start_day", "value"),
    Input("search_start_month", "value"),
    Input("search_days", "value"),
    Input("wet_threshold", "value"),
    Input("running_days", "value"),
    Input("running_total", "value"),
    Input("min_rainy_days", "value"),
    Input("dry_days", "value"),
    Input("dry_spell", "value"),
    Input("cess_start_day", "value"),
    Input("cess_start_month", "value"),
    Input("cess_search_days", "value"),
    Input("cess_soil_moisture", "value"),
    Input("cess_dry_spell", "value"),
)
def length_plots(
    marker_pos,
    search_start_day,
    search_start_month,
    search_days,
    wet_threshold,
    running_days,
    running_total,
    min_rainy_days,
    dry_days,
    dry_spell,
    cess_start_day,
    cess_start_month,
    cess_search_days,
    cess_soil_moisture,
    cess_dry_spell,
):
    if not CONFIG["ison_cess_date_hist"]:
        tab_style = {"display": "none"}
        return {}, {}, tab_style
    else:
        tab_style = {}
        lat = marker_pos[0]
        lng = marker_pos[1]
        try:
            precip = pingrid.sel_snap(rr_mrg.precip, lat, lng)
            isnan = np.isnan(precip).any()
            if isnan:
                error_fig = pingrid.error_fig(error_msg="Data missing at this location")
                germ_sentence = ""
                return error_fig, error_fig, tab_style
        except KeyError:
            error_fig = pingrid.error_fig(error_msg="Grid box out of data domain")
            germ_sentence = ""
            return error_fig, error_fig, tab_style
        precip.load()
        try:
            onset_delta = calc.seasonal_onset_date(
                precip,
                int(search_start_day),
                calc.strftimeb2int(search_start_month),
                int(search_days),
                int(wet_threshold),
                int(running_days),
                int(running_total),
                int(min_rainy_days),
                int(dry_days),
                int(dry_spell),
                time_dim="T",
            )
            isnan = np.isnan(onset_delta["onset_delta"]).all()
            if isnan:
                error_fig = pingrid.error_fig(error_msg="No onset dates were found")
                return error_fig, error_fig, tab_style
        except TypeError:
            error_fig = pingrid.error_fig(
                error_msg="Please ensure all onset input boxes are filled."
            )
            return error_fig, error_fig, tab_style
        try:
            cess_delta = calc.seasonal_cess_date_from_rain(
                precip,
                int(cess_start_day),
                calc.strftimeb2int(cess_start_month),
                int(cess_search_days),
                int(cess_soil_moisture),
                int(cess_dry_spell),
                5,
                60,
                60./3.,
                time_dim="T",
            )
            isnan = np.isnan(cess_delta["cess_delta"]).all()
            if isnan:
                error_fig = pingrid.error_fig(error_msg="No cessation dates were found")
                return error_fig, error_fig, tab_style
        except TypeError:
            error_fig = pingrid.error_fig(error_msg="Please ensure all cessation input boxes are filled")
            return error_fig, error_fig, tab_style
        if cess_delta["T"][0] < onset_delta["T"][0]:
            cess_delta = cess_delta.isel({"T": slice(1, None)})
        if cess_delta["T"].size != onset_delta["T"].size:
            onset_delta = onset_delta.isel({"T": slice(None, -1)})
        try:
            seasonal_length = (
                (cess_delta["T"] + cess_delta["cess_delta"]).drop_indexes("T")
                - (onset_delta["T"] + onset_delta["onset_delta"]).drop_indexes("T")
            )
            isnan = np.isnan(seasonal_length).all()
            if isnan:
                error_fig = pingrid.error_fig(error_msg="Onset or cessation not found for any season")
                return error_fig, error_fig, tab_style
        except TypeError:
            error_fig = pingrid.error_fig(error_msg="Please ensure all onset/cessation input boxes are filled")
            return error_fig, error_fig, tab_style
        length_graph = pgo.Figure()
        length_graph.add_trace(
            pgo.Bar(
                x=onset_delta["T"].dt.year.values,
                y=seasonal_length.squeeze().dt.days.values,
                customdata=np.stack((
                    (onset_delta["T"] + onset_delta["onset_delta"]).dt.strftime("%-d %b %Y"),
                    (cess_delta["T"] + cess_delta["cess_delta"]).dt.strftime("%-d %b %Y"),
                ), axis=-1),
                hovertemplate="%{customdata[0]} to %{customdata[1]}",
                name="",
            )
        )
        length_graph.update_layout(
            xaxis_title="Year",
            yaxis_title=f"Season Length in days",
            title=f"Season Length at ({round_latLng(lat)}N,{round_latLng(lng)}E)",
        )
        quantiles = np.arange(1, seasonal_length["T"].size + 1) / (seasonal_length["T"].size + 1)
        length_quant = seasonal_length.dt.days.quantile(quantiles, dim="T").squeeze()
        cdf_graph = pgo.Figure()
        cdf_graph.add_trace(
            pgo.Scatter(
                x=length_quant.values,
                y=(1 - quantiles),
                hovertemplate="%{y:.0%} chance of exceeding %{x:d} days",
                name="",
                line=pgo.scatter.Line(color="blue"),
            )
        )
        cdf_graph.update_traces(mode="lines", connectgaps=False)
        cdf_graph.update_layout(
            xaxis_title=f"Season Length in days",
            yaxis_title="Probability of exceeding",
        )
        return length_graph, cdf_graph, tab_style


@FLASK.route(f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>")
def onset_tile(tz, tx, ty):
    parse_arg = pingrid.parse_arg
    map_choice = parse_arg("map_choice")
    search_start_day = parse_arg("search_start_day", int)
    search_start_month1 = parse_arg("search_start_month", calc.strftimeb2int)
    search_days = parse_arg("search_days", int)
    wet_thresh = parse_arg("wet_thresh", float)
    wet_spell_length = parse_arg("wet_spell_length", int)
    wet_spell_thresh = parse_arg("wet_spell_thresh", float)
    min_wet_days = parse_arg("min_wet_days", int)
    dry_spell_length = parse_arg("dry_spell_length", int)
    dry_spell_search = parse_arg("dry_spell_search", int)
    cess_start_day = parse_arg("cess_start_day", int)
    cess_start_month1 = parse_arg("cess_start_month", calc.strftimeb2int)
    cess_search_days = parse_arg("cess_search_days", int)
    cess_soil_moisture = parse_arg("cess_soil_moisture", float)
    cess_dry_spell = parse_arg("cess_dry_spell", int)
    prob_exc_thresh_onset = parse_arg("prob_exc_thresh_onset", int)
    prob_exc_thresh_length = parse_arg("prob_exc_thresh_length", int)
    prob_exc_thresh_tot = parse_arg("prob_exc_thresh_tot", int)

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
        return pingrid.image_resp(pingrid.empty_tile())

    if map_choice == "monit":
        precip_tile = rr_mrg.precip.isel({"T": slice(-366, None)})
        search_start_dm = calc.sel_day_and_month(precip_tile["T"], search_start_day, search_start_month1)
        precip_tile = precip_tile.sel({"T": slice(search_start_dm.values[0], None)})
    else:
        precip_tile = rr_mrg.precip

    precip_tile = precip_tile.sel(
        X=slice(x_min - x_min % RESOLUTION, x_max + RESOLUTION - x_max % RESOLUTION),
        Y=slice(y_min - y_min % RESOLUTION, y_max + RESOLUTION - y_max % RESOLUTION),
    ).compute()

    map_min = np.timedelta64(0) if map_choice in ["monit", "mean"] else 0
    colormap = CMAPS["rainbow"]

    if map_choice == "monit":
        map_data = calc.onset_date(
            precip_tile,
            wet_thresh,
            wet_spell_length,
            wet_spell_thresh,
            min_wet_days,
            dry_spell_length,
            0
        )
        map_max = np.timedelta64((precip_tile["T"][-1] - precip_tile["T"][0]).values, 'D')
    else:
        onset_dates = calc.seasonal_onset_date(
            precip_tile,
            search_start_day,
            search_start_month1,
            search_days,
            wet_thresh,
            wet_spell_length,
            wet_spell_thresh,
            min_wet_days,
            dry_spell_length,
            dry_spell_search,
        )
        if ("length" in map_choice) | ("total" in map_choice):
            cess_dates = calc.seasonal_cess_date_from_rain(
                precip_tile,
                cess_start_day,
                cess_start_month1,
                cess_search_days,
                cess_soil_moisture,
                cess_dry_spell,
                5,
                60,
                60./3.,
            )
            if "length" in map_choice:
                if cess_dates["T"][0] < onset_dates["T"][0]:
                    cess_dates = cess_dates.isel({"T": slice(1, None)})
                if cess_dates["T"].size != onset_dates["T"].size:
                    onset_dates = onset_dates.isel({"T": slice(None, -1)})
                seasonal_length = (
                    (cess_dates["T"] + cess_dates["cess_delta"]).drop_indexes("T")
                    - (onset_dates["T"] + onset_dates["onset_delta"]).drop_indexes("T")
                ) #.astype("timedelta64[D]")
        if map_choice == "mean":
            map_data = onset_dates.onset_delta.mean("T")
            map_max = np.timedelta64(search_days, 'D')
        if map_choice == "stddev":
            map_data = onset_dates.onset_delta.dt.days.std(dim="T", skipna=True)
            map_max = int(search_days/3)
        if map_choice == "pe":
            map_data = (
                onset_dates.onset_delta.fillna(
                    np.timedelta64(search_days+1, 'D')
                ) > np.timedelta64(prob_exc_thresh_onset, 'D')
            ).mean("T") * 100
            map_max = 100
            colormap = CMAPS["correlation"]
        if map_choice == "length_mean":
            map_data = seasonal_length.mean("T")
            map_max = np.timedelta64(int(CONFIG["map_text"][map_choice]["map_max"]), 'D')
        if map_choice == "length_stddev":
            map_data = seasonal_length.dt.days.std(dim="T", skipna=True)
            map_max = CONFIG["map_text"][map_choice]["map_max"]
        if map_choice == "length_pe":
            map_data = (seasonal_length < np.timedelta64(prob_exc_thresh_length, 'D')).mean("T") * 100
            map_max = 100
            colormap = CMAPS["correlation"]
    map_data.attrs["colormap"] = colormap
    map_data = map_data.rename(X="lon", Y="lat")
    map_data.attrs["scale_min"] = map_min
    map_data.attrs["scale_max"] = map_max
    result = pingrid.tile(map_data, tx, ty, tz, clip_shape)
    return result


@APP.callback(
    Output("colorbar", "colorscale"),
    Output("colorbar", "max"),
    Output("colorbar", "tickValues"),
    Output("colorbar", "unit"),
    Input("search_start_day", "value"),
    Input("search_start_month", "value"),
    Input("search_days", "value"),
    Input("map_choice", "value")
)
def set_colorbar(search_start_day, search_start_month, search_days, map_choice):
    colorbar = CMAPS["rainbow"].to_dash_leaflet()
    if "pe" in map_choice:
        colorbar = CMAPS["correlation"].to_dash_leaflet()
        tick_freq = 10
        map_max = 100
        unit = "%"
    if map_choice == "mean":
        tick_freq = 10
        map_max = int(search_days)
        unit = "days" 
    if map_choice == "stddev":
        tick_freq = 5
        map_max = int(int(search_days)/3)
        unit = "days"
    if map_choice == "length_mean":
        tick_freq = 20
        map_max = CONFIG["map_text"][map_choice]["map_max"]
        unit = "days"
    if map_choice == "length_stddev":
        tick_freq = 5
        map_max = CONFIG["map_text"][map_choice]["map_max"]
        unit = "days"
    if map_choice == "monit":
        precip = rr_mrg.precip.isel({"T": slice(-366, None)})
        search_start_dm = calc.sel_day_and_month(
            precip["T"], int(search_start_day), calc.strftimeb2int(search_start_month)
        )
        map_max = np.timedelta64(
            (precip["T"][-1] - search_start_dm).values[0], 'D'
        ).astype(int)
        tick_freq = 10
        unit = "days"
    return colorbar, map_max, [i for i in range(0, map_max + 1) if i % tick_freq == 0], unit
