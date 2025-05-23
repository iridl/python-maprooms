import os
import flask
import dash
from dash import ALL
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import dash_leaflet as dlf
from pathlib import Path
import pingrid 
from pingrid import CMAPS
from . import layout_monit
import calc
import maproom_utilities as mapr_u
import plotly.graph_objects as pgo
import pandas as pd
import numpy as np
import urllib
import datetime
from controls import Sentence, DateNoYear, Number, Select, Text

import xarray as xr
import agronomy as ag

from globals_ import GLOBAL_CONFIG, FLASK

CONFIG = GLOBAL_CONFIG["maprooms"]["wat_bal"]

PRECIP_PARAMS = {
    "variable": "precip", "time_res": "daily", "ds_conf": GLOBAL_CONFIG["datasets"]
}

def register(FLASK, config):

    PFX = f'{GLOBAL_CONFIG["url_path_prefix"]}/{config["core_path"]}'
    TILE_PFX = f"{PFX}/tile"
    API_WINDOW = 7
    STD_TIME_FORMAT = "%Y-%m-%d"
    HUMAN_TIME_FORMAT = "%-d %b %Y"

    APP = dash.Dash(
        __name__,
        server=FLASK,
        external_stylesheets=[dbc.themes.BOOTSTRAP,],
        url_base_pathname=f"{PFX}/",
        meta_tags=[
            {"name": "description", "content": "Water Balance Maproom"},
            {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
        ],
    )
    APP.title = config["title"]
    APP.layout = layout_monit.app_layout()

    @APP.callback(
        Output("lat_input", "min"),
        Output("lat_input", "max"),
        Output("lat_input_tooltip", "children"),
        Output("lng_input", "min"),
        Output("lng_input", "max"),
        Output("lng_input_tooltip", "children"),
        Output("map", "center"),
        Output("map", "max_bounds"),
        Output("navbarbrand", "children"),
        Output("app_title", "children"),
        Output("maplabdesc", "children"),
        Output("mapchoice", "children"),
        Output("curseas", "children"),
        Output("otherseas", "children"),
        Input("location", "pathname"),
    )
    def initialize(path):
        rr_mrg = calc.get_data(**PRECIP_PARAMS)
        center_of_the_map = [
            ((rr_mrg["Y"][int(rr_mrg["Y"].size/2)].values)),
            ((rr_mrg["X"][int(rr_mrg["X"].size/2)].values)),
        ]
        lat_res = (rr_mrg["Y"][0 ]- rr_mrg["Y"][1]).values
        lat_min = str((rr_mrg["Y"][-1] - lat_res/2).values)
        lat_max = str((rr_mrg["Y"][0] + lat_res/2).values)
        lon_res = (rr_mrg["X"][1] - rr_mrg["X"][0]).values
        lon_min = str((rr_mrg["X"][0] - lon_res/2).values)
        lon_max = str((rr_mrg["X"][-1] + lon_res/2).values)
        lat_label = lat_min + " to " + lat_max + " by " + str(lat_res) + "˚"
        lon_label = lon_min + " to " + lon_max + " by " + str(lon_res) + "˚"
        first_year =  rr_mrg["T"][0].dt.year.values
        one_to_last_year = rr_mrg["T"][-367].dt.year.values
        last_year =  rr_mrg["T"][-1].dt.year.values
        map_label_description = [
            html.P([html.H6(val["menu_label"]), html.P(val["description"])])
                for key, val in config["map_text"].items()
        ]
        map_choice = Select(
            "map_choice",
            [key for key, val in config["map_text"].items()],
            labels=[val["menu_label"] for key, val in config["map_text"].items()],
        )
        current_season = (
            Sentence(
                "Planting Date",
                DateNoYear("planting_", 1, config["planting_month"]),
                "for",
                Text("crop_name", config["crop_name"]),
                "crop cultivars: initiated at",
            ),
            Sentence(
                Number("kc_init", config["kc_v"][0], min=0, max=2, width="5em"),
                "through",
                Number(
                    "kc_init_length", config["kc_l"][0], min=0, max=99, width="4em"
                ),
                "days of initialization to",
            ),
            Sentence(
                Number("kc_veg", config["kc_v"][1], min=0, max=2, width="5em"),
                "through",
                Number(
                    "kc_veg_length", config["kc_l"][1], min=0, max=99, width="4em"
                ),
                "days of growth to",
            ),
            Sentence(
                Number("kc_mid", config["kc_v"][2], min=0, max=2, width="5em"),
                "through",
                Number(
                    "kc_mid_length", config["kc_l"][2], min=0, max=99, width="4em"
                ),
                "days of mid-season to",
            ),
            Sentence(
                Number("kc_late", config["kc_v"][3], min=0, max=2, width="5em"),
                "through",
                Number(
                    "kc_late_length", config["kc_l"][3], min=0, max=99, width="4em"
                ),
                "days of late-season to",
            ),
            Sentence(
                Number("kc_end", config["kc_v"][4], min=0, max=2, width="5em"),
            ),
        )
        other_season = (
            Sentence(
                "Planting Date",
                DateNoYear("planting2_", 1, config["planting_month"]),
                "",
                Number(
                    "planting2_year",
                    one_to_last_year,
                    min=f'{first_year}',
                    max=f'{last_year}',
                    width="120px",
                ),
            ),
            Sentence(
                "for",
                Text("crop2_name", config["crop_name"]),
                "crop cultivars: initiated at",
            ),
            Sentence(
                Number("kc2_init", config["kc_v"][0], min=0, max=2, width="5em"),
                "through",
                Number(
                    "kc2_init_length", config["kc_l"][0], min=0, max=99, width="4em"
                ),
                "days of initialization to",
            ),
            Sentence(
                Number("kc2_veg", config["kc_v"][1], min=0, max=2, width="5em"),
                "through",
                Number(
                    "kc2_veg_length", config["kc_l"][1], min=0, max=99, width="4em"
                ),
                "days of growth to",
            ),
            Sentence(
                Number("kc2_mid", config["kc_v"][2], min=0, max=2, width="5em"),
                "through",
                Number(
                    "kc2_mid_length", config["kc_l"][2], min=0, max=99, width="4em"
                ),
                "days of mid-season to",
            ),
            Sentence(
                Number("kc2_late", config["kc_v"][3], min=0, max=2, width="5em"),
                "through",
                Number(
                    "kc2_late_length", config["kc_l"][3], min=0, max=99, width="4em"
                ),
                "days of late-season to",
            ),
            Sentence(
                Number("kc2_end", config["kc_v"][4], min=0, max=2, width="5em"),
            ),
        )
        return (
            lat_min, lat_max, lat_label,
            lon_min, lon_max, lon_label,
            center_of_the_map,
            [[lat_min, lon_min],[lat_max, lon_max]],
            ("Climate and Agriculture / " + config["title"]),
            config["title"], map_label_description, map_choice,
            current_season, other_season,
        )


    @APP.callback(
        Output("time_selection", "options"),
        Output("time_selection", "value"),
        Input("planting_day","value"),
        Input("planting_month", "value"),
        Input("wat_bal_plot", "clickData"),
        State("time_selection", "options"),
    )
    def update_time_sel(planting_day, planting_month, graph_click, current_options):
        if dash.ctx.triggered_id == "wat_bal_plot":
            time_options = current_options
            the_value = graph_click["points"][0]["x"]
        else:
            rr_mrg = calc.get_data(**PRECIP_PARAMS)
            time_range = rr_mrg["T"].isel({"T": slice(-366, None)})
            p_d = calc.sel_day_and_month(
                time_range, int(planting_day), calc.strftimeb2int(planting_month)
            ).squeeze()
            time_range = time_range.where(
                time_range >= p_d, drop=True
            )
            time_options = [
                {
                    "label": tr.dt.strftime(HUMAN_TIME_FORMAT).values,
                    "value": tr.dt.strftime(STD_TIME_FORMAT).values,
                }
                for tr in time_range
            ]
            the_value = time_options[-1]["value"]
        return time_options, the_value


    @APP.callback(
        Output("layers_control", "children"),
        Input("map_choice", "value"),
        Input("time_selection", "value"),
        Input("submit_kc", "n_clicks"),
        State("planting_day", "value"),
        State("planting_month", "value"),
        State("kc_init", "value"),
        State("kc_init_length", "value"),
        State("kc_veg", "value"),
        State("kc_veg_length", "value"),
        State("kc_mid", "value"),
        State("kc_mid_length", "value"),
        State("kc_late", "value"),
        State("kc_late_length", "value"),
        State("kc_end", "value"),
    )
    def make_map(
        map_choice,
        the_date,
        n_clicks,
        planting_day,
        planting_month,
        kc_init,
        kc_init_length,
        kc_veg,
        kc_veg_length,
        kc_mid,
        kc_mid_length,
        kc_late,
        kc_late_length,
        kc_end,
    ):
        qstr = urllib.parse.urlencode({
            "map_choice": map_choice,
            "the_date": the_date,
            "n_clicks": n_clicks,
            "planting_day": planting_day,
            "planting_month": planting_month,
            "kc_init": kc_init,
            "kc_init_length": kc_init_length,
            "kc_veg": kc_veg,
            "kc_veg_length": kc_veg_length,
            "kc_mid": kc_mid,
            "kc_mid_length": kc_mid_length,
            "kc_late": kc_late,
            "kc_late_length": kc_late_length,
            "kc_end": kc_end,
        })
        return [
            dlf.BaseLayer(
                dlf.TileLayer(url=(
                    "https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"
                ),),
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
            mapr_u.make_adm_overlay(
                adm_name=adm["name"],
                adm_geojson=calc.geom2GeoJSON(
                    calc.get_geom(level=i, conf=GLOBAL_CONFIG)
                ),
                adm_color=adm["color"],
                adm_lev=i+1,
                adm_weight=len(GLOBAL_CONFIG["datasets"]["shapes_adm"])-i,
                is_checked=adm["is_checked"],
            )
            for i, adm in enumerate(GLOBAL_CONFIG["datasets"]["shapes_adm"])
        ] + [
            dlf.Overlay(
                dlf.TileLayer(
                    url=f"{TILE_PFX}/{{z}}/{{x}}/{{y}}?{qstr}",
                    opacity=1,
                ),
                name="Water_Balance",
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
        Output("map_title", "children"),
        Input("map_choice", "value"),
        Input("crop_name", "value"),
    )
    def write_map_title(map_choice, crop_name):
        return f"{config['map_text'][map_choice]['menu_label']} for {crop_name}"


    @APP.callback(
        Output("map_description", "children"),
        Input("map_choice", "value"),
    )
    def write_map_description(map_choice):
        return config["map_text"][map_choice]["description"]    


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
        rr_mrg = calc.get_data(**PRECIP_PARAMS)
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


    def wat_bal(
        precip,
        et,
        taw,
        planting_day,
        planting_month,
        kc_init_length,
        kc_veg_length,
        kc_mid_length,
        kc_late_length,
        kc_init,
        kc_veg,
        kc_mid,
        kc_late,
        kc_end,
        planting_year=None,
        the_date=None,
        time_coord="T",
    ):
        kc_periods = pd.TimedeltaIndex(
            [0, kc_init_length, kc_veg_length, kc_mid_length, kc_late_length],
            unit="D",
        )
        kc_params = xr.DataArray(data=[
            kc_init, kc_veg, kc_mid, kc_late, kc_end
        ], dims=["kc_periods"], coords=[kc_periods])
    
        p_d = calc.sel_day_and_month(
            precip[time_coord], planting_day, planting_month
        )
        p_d = (p_d[-1] if planting_year is None else p_d.where(
            p_d.dt.year == planting_year, drop=True
        )).squeeze(drop=True).rename("p_d")

        if the_date is None:
            the_date = (
                p_d + np.timedelta64(365, "D")
            ).dt.strftime(HUMAN_TIME_FORMAT)
        precip = precip.sel(T=slice(
            (
                p_d - np.timedelta64(API_WINDOW - 1, "D")
            ).dt.strftime(HUMAN_TIME_FORMAT),
            the_date,
        ))
        precip_effective = (
            precip.isel({"T": slice(API_WINDOW - 1, None)})
            - ag.api_runoff(
                precip.isel({"T": slice(API_WINDOW - 1, None)}),
                api = ag.antecedent_precip_ind(precip, API_WINDOW),
            )
        )
        return ag.soil_plant_water_balance(
            precip_effective,
            et=et,
            taw=taw,
            sminit=taw/3.,
            kc_params=kc_params,
            planting_date=p_d,
        )
    

    def wat_bal_ts(
        precip,
        map_choice,
        et,
        taw,
        planting_day,
        planting_month,
        kc_init_length,
        kc_veg_length,
        kc_mid_length,
        kc_late_length,
        kc_init,
        kc_veg,
        kc_mid,
        kc_late,
        kc_end,
        planting_year=None,
        time_coord="T",
    ):
        try:
            water_balance_outputs = wat_bal(
                precip, et, taw,
                planting_day, planting_month,
                kc_init_length,
                kc_veg_length,
                kc_mid_length,
                kc_late_length,
                kc_init, kc_veg, kc_mid, kc_late, kc_end,
                planting_year=planting_year,
                time_coord=time_coord,
            )
            for wbo in water_balance_outputs:
                if map_choice == "paw" and wbo.name == "sm":
                    ts = 100 * wbo / taw
                elif map_choice == "water_excess" and wbo.name == "sm":
                    ts = xr.DataArray(
                        np.isclose(wbo, taw).cumsum(),
                        dims="T",
                        coords={"T": wbo["T"]},
                    )
                elif map_choice == "peff":
                    ts = precip_effective
                elif (wbo.name == map_choice):
                    ts = wbo
        except TypeError:
            #Later tested to return error image rather than broken one
            ts = None
        return ts


    def plot_scatter(ts, name, color, dash=None, customdata=None):
        hovertemplate = (
            "%{y} on %{x|"+HUMAN_TIME_FORMAT+"}"
                if customdata is None
                else "%{y} on %{customdata|"+HUMAN_TIME_FORMAT+"}"
        )
        return pgo.Scatter(
            x=ts["T"].dt.strftime(STD_TIME_FORMAT),
            y=ts.values,
            customdata=customdata,
            hovertemplate=hovertemplate,
            name=name,
            line=pgo.scatter.Line(color=color, dash=dash),
            connectgaps=False,
        )

    @APP.callback(
        Output("wat_bal_plot", "figure"),
        Input("loc_marker", "position"),
        Input("map_choice", "value"),
        Input("submit_kc", "n_clicks"),
        Input("submit_kc2", "n_clicks"),
        State("planting_day", "value"),
        State("planting_month", "value"),
        State("crop_name", "value"),
        State("kc_init", "value"),
        State("kc_init_length", "value"),
        State("kc_veg", "value"),
        State("kc_veg_length", "value"),
        State("kc_mid", "value"),
        State("kc_mid_length", "value"),
        State("kc_late", "value"),
        State("kc_late_length", "value"),
        State("kc_end", "value"),
        State("planting2_day", "value"),
        State("planting2_month", "value"),
        State("planting2_year", "value"),
        State("crop2_name", "value"),
        State("kc2_init", "value"),
        State("kc2_init_length", "value"),
        State("kc2_veg", "value"),
        State("kc2_veg_length", "value"),
        State("kc2_mid", "value"),
        State("kc2_mid_length", "value"),
        State("kc2_late", "value"),
        State("kc2_late_length", "value"),
        State("kc2_end", "value"),
    )
    def wat_bal_plots(
        marker_pos,
        map_choice,
        n_clicks,
        n2_clicks,
        planting_day,
        planting_month,
        crop_name,
        kc_init,
        kc_init_length,
        kc_veg,
        kc_veg_length,
        kc_mid,
        kc_mid_length,
        kc_late,
        kc_late_length,
        kc_end,
        planting2_day,
        planting2_month,
        planting2_year,
        crop2_name,
        kc2_init,
        kc2_init_length,
        kc2_veg,
        kc2_veg_length,
        kc2_mid,
        kc2_mid_length,
        kc2_late,
        kc2_late_length,
        kc2_end,
    ):
        rr_mrg = calc.get_data(**PRECIP_PARAMS)
        first_year = rr_mrg["T"][0].dt.year.values
        last_year = rr_mrg["T"][-1].dt.year.values
        if planting2_year is None:
            return pingrid.error_fig(error_msg=(
                f"Planting date must be between {first_year} and {last_year}"
            ))
        lat = marker_pos[0]
        lng = marker_pos[1]
        try:
            taw = pingrid.sel_snap(
                calc.get_taw(GLOBAL_CONFIG["datasets"]), lat, lng
            )
        except KeyError:
            return pingrid.error_fig(error_msg="Grid box out of data domain")
        precip = pingrid.sel_snap(rr_mrg, lat, lng)
        if np.isnan(precip).all():
            return pingrid.error_fig(error_msg="Data missing at this location")

        ts = wat_bal_ts(
            precip,
            map_choice,
            5,
            taw,
            int(planting_day),
            calc.strftimeb2int(planting_month),
            int(kc_init_length),
            int(kc_veg_length),
            int(kc_mid_length),
            int(kc_late_length),
            float(kc_init),
            float(kc_veg),
            float(kc_mid),
            float(kc_late),
            float(kc_end),
        )
        if (ts is None):
            return pingrid.error_fig(error_msg=(
                f"Please ensure all input boxes are filled for the calculation "
                f"to run."
            ))
        ts2 = wat_bal_ts(
            precip,
            map_choice,
            5,
            taw,
            int(planting2_day),
            calc.strftimeb2int(planting2_month),
            int(kc2_init_length),
            int(kc2_veg_length),
            int(kc2_mid_length),
            int(kc2_late_length),
            float(kc2_init),
            float(kc2_veg),
            float(kc2_mid),
            float(kc2_late),
            float(kc2_end),
            planting_year=int(planting2_year)
        )
        if (ts2 is None):
            return pingrid.error_fig(error_msg=(
                f"Please ensure all input boxes are filled for the calculation "
                f"to run."
            ))
        #Save actual start of ts2
        ts2_start = ts2["T"][0]
        #Find corresponding day closer to ts
        p_d2 = calc.sel_day_and_month(
            precip["T"], int(planting2_day), calc.strftimeb2int(planting2_month)
        )
        p_d2 = p_d2.where(
            abs(ts["T"][0] - p_d2) == abs(ts["T"][0] - p_d2).min(), drop=True
        ).squeeze(drop=True).rename("p_d2")
        #Assign ts-contemporary dates to ts2
        ts2 = ts2.assign_coords({"T": pd.date_range(datetime.datetime(
            p_d2.dt.year.values, p_d2.dt.month.values, p_d2.dt.day.values
        ), periods=ts2["T"].size)})
        #Align ts and ts2 so that they have same size
        #(otherwise Scatter goes bananas)
        ts, ts2 = xr.align(ts, ts2, join="outer")
        #Recreate real dates of ts2 after alignment in ts contemporary dates
        #to feed to customdata to hovertemplate
        ts2_customdata = calc.sel_day_and_month(
            precip["T"], ts2["T"][0].dt.day, ts2["T"][0].dt.month
        )
        ts2_customdata = ts2_customdata.where(
            abs(ts2_start - ts2_customdata) == abs(ts2_start - ts2_customdata).min(),
            drop=True,
        )[0].squeeze(drop=True)
        ts2_customdata = pd.date_range(datetime.datetime(
            ts2_customdata.dt.year.values,
            ts2_customdata.dt.month.values,
            ts2_customdata.dt.day.values,
        ), periods=ts2["T"].size).strftime(STD_TIME_FORMAT)
        wat_bal_graph = pgo.Figure()
        wat_bal_graph.add_trace(plot_scatter(ts, "Current", "green"))
        wat_bal_graph.add_trace(plot_scatter(
            ts2, "Comparison", "blue", dash="dash", customdata=ts2_customdata
        ))
        wat_bal_graph.update_layout(
            xaxis_title="Time",
            xaxis_tickformat="%-d %b",
            yaxis_title=(
                f"{config['map_text'][map_choice]['menu_label']} "
                f"[{config['map_text'][map_choice]['units']}]"
            ),
            title=(
                f"{config['map_text'][map_choice]['menu_label']} for {crop_name} "
                f"at ({round_latLng(lat)}N,{round_latLng(lng)}E)"
            ),
        )
        return wat_bal_graph


    @FLASK.route(f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>")
    def wat_bal_tile(tz, tx, ty):
        parse_arg = pingrid.parse_arg
        map_choice = parse_arg("map_choice")
        the_date = parse_arg("the_date", str)
        planting_day = parse_arg("planting_day", int)
        planting_month1 = parse_arg("planting_month", calc.strftimeb2int)
        kc_init = parse_arg("kc_init", float)
        kc_init_length = parse_arg("kc_init_length", int)
        kc_veg = parse_arg("kc_init", float)
        kc_veg_length = parse_arg("kc_init_length", int)
        kc_mid = parse_arg("kc_mid", float)
        kc_mid_length = parse_arg("kc_mid_length", int)
        kc_late = parse_arg("kc_late", float)
        kc_late_length = parse_arg("kc_late_length", int)
        kc_end = parse_arg("kc_end", float)

        rr_mrg = calc.get_data(**PRECIP_PARAMS)
        precip = rr_mrg
        x_min = pingrid.tile_left(tx, tz)
        x_max = pingrid.tile_left(tx + 1, tz)
        # row numbers increase as latitude decreases
        y_max = pingrid.tile_top_mercator(ty, tz)
        y_min = pingrid.tile_top_mercator(ty + 1, tz)
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
        _, taw = xr.align(
            precip,
            calc.get_taw(GLOBAL_CONFIG["datasets"]),
            join="override",
            exclude="T",
        )
        # Assumes that grid spacing is regular and cells are square. When we
        # generalize this, don't make those assumptions.
        resolution = rr_mrg['X'][1].item() - rr_mrg['X'][0].item()
        # The longest possible distance between a point and the center of the
        # grid cell containing that point.
        taw_tile = taw.sel(
            X=slice(
                x_min - x_min % resolution, x_max + resolution - x_max % resolution
            ),
            Y=slice(
                y_min - y_min % resolution, y_max + resolution - y_max % resolution
            ),
        ).compute()
        precip_tile = precip.sel(
            X=slice(
                x_min - x_min % resolution, x_max + resolution - x_max % resolution
            ),
            Y=slice(
                y_min - y_min % resolution, y_max + resolution - y_max % resolution
            ),
        ).compute()
        sm, drainage, et_crop, et_crop_red, planting_date = wat_bal(
            precip_tile,
            5,
            taw_tile,
            planting_day,
            planting_month1,
            kc_init_length,
            kc_veg_length,
            kc_mid_length,
            kc_late_length,
            kc_init,
            kc_veg,
            kc_mid,
            kc_late,
            kc_end,
            the_date=the_date,
        )
        map_max = config["taw_max"]
        if map_choice == "sm":
            map = sm
        elif map_choice == "drainage":
            map = drainage
        elif map_choice == "et_crop":
            map = et_crop
        elif map_choice == "paw":
            map = 100 * sm / taw_tile
            map_max = 100
        elif map_choice == "water_excess":
            #this is to accommodate pingrid tiling
            #because NaN == 1 is False thus 0
            #but tiling doesn't like all 0s on presumably empty tiles
            #instead it wants all NaNs, what this does.
            #It's ok because sum of all NaNs is NaN
            #while sum with some NaNs treats them as 0.
            #which is what we want: count of days where sm == taw
            map = (sm / taw_tile).where(lambda x: np.isclose(x, 1)).sum(dim="T")
            map_max = sm["T"].size
        elif map_choice == "peff":
            map = precip_effective
            map_max = config["peff_max"]
        else:
            raise Exception("can not enter here")
        map = map.isel(T=-1, missing_dims='ignore')
        map.attrs["colormap"] = CMAPS["precip"]
        map = map.rename(X="lon", Y="lat")
        map.attrs["scale_min"] = 0
        map.attrs["scale_max"] = map_max
        clip_shape = calc.get_geom(level=0, conf=GLOBAL_CONFIG)["the_geom"][0]
        return pingrid.tile(map, tx, ty, tz, clip_shape)


    @APP.callback(
        Output("colorbar", "colorscale"),
        Output("colorbar", "max"),
        Output("colorbar", "tickValues"),
        Output("colorbar", "unit"),
        Input("map_choice", "value"),
        Input("time_selection", "value"),
        State("planting_day", "value"),
        State("planting_month", "value"),
    )
    def set_colorbar(map_choice, the_date, planting_day, planting_month):
        if map_choice == "paw":
            map_max = 100
        elif map_choice == "water_excess":
            rr_mrg = calc.get_data(**PRECIP_PARAMS)
            time_range = rr_mrg["T"][-366:]
            p_d = calc.sel_day_and_month(
                time_range, int(planting_day), calc.strftimeb2int(planting_month)
            ).squeeze(drop=True).rename("p_d")
            map_max = time_range.sel(
                T=slice(p_d.dt.strftime(HUMAN_TIME_FORMAT), the_date)
            ).size
        elif map_choice == "peff":
            map_max = config["peff_max"]
        else:
            map_max = config["taw_max"]
        return (
            CMAPS["precip"].to_dash_leaflet(),
            map_max,
            [i for i in range(0, map_max + 1) if i % int(map_max/8) == 0],
            config['map_text'][map_choice]['units'],
        )
