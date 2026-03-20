import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import pingrid
from pingrid import CMAPS
from . import layout
import plotly.graph_objects as pgo
import plotly.express as px
import numpy as np
import xarray as xr
import terc_fcst_calc as tfc
import maproom_utilities as mru
from globals_ import GLOBAL_CONFIG
from fieldsets import Select
import datetime
import predictions


def register(FLASK, config):
    PFX = f"{GLOBAL_CONFIG['url_path_prefix']}/{config['core_path']}"
    TILE_PFX = f"{PFX}/tile"

    # App

    APP = dash.Dash(
        __name__,
        server=FLASK,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
        ],
        url_base_pathname=f"{PFX}/",
        meta_tags=[
            {"name": "description", "content": "Forecast"},
            {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
        ],
    )
    APP.title = "Tercile Forecast"

    APP.layout = layout.app_layout()

    @APP.callback(
            Output("start_div", "children"),
            Output("lead_div", "children"),
            Output("below_cb", "colorscale"),
            Output("above_cb", "colorscale"),
            Output("lat_input", "min"),
            Output("lat_input", "max"),
            Output("lat_input_tooltip", "children"),
            Output("lng_input", "min"),
            Output("lng_input", "max"),
            Output("lng_input_tooltip", "children"),
            Output("map", "center"),
            Input("location", "pathname"),
    )
    def initialize(path):
        S_dates = tfc.get_issues(config)
        issue_select = Select("start_date", options=[d.strftime("%b %Y") for d in S_dates])
        fcst_ds = tfc.get_fcst(config)
        target_dict = tfc.get_targets_dict(fcst_ds, S_dates[0])
        lead_select = Select("lead_time",
            options=[ld["value"] for ld in target_dict],
            labels=[ld["label"] for ld in target_dict],
        )
        return (
            issue_select, lead_select,
            CMAPS["prcp_terciles_below"].to_dash_leaflet(),
            CMAPS["prcp_terciles_above"].to_dash_leaflet(),
        ) + mru.initialize_map(fcst_ds)


    @APP.callback(
        Output("lead_time","options"),
        Input("start_date","value"),
    )
    def target_range_options(start_date):
        S = datetime.datetime(
            int(start_date[4:8]), predictions.strftimeb2int(start_date[0:3]), 1
        )
        return tfc.get_targets_dict(tfc.get_fcst(config, start_date), S)


    @APP.callback(
        Output("map_title", "children"),
        Input("start_date", "value"),
        Input("lead_time", "value"),
        Input("lead_time", "options"),
    )
    def write_map_title(start_date, lead_time, targets):
        for option in targets :
            if option["value"] == int(lead_time) :
                target = option["label"]
        return f'{target} {config["variable"]} Forecast issued {start_date}'


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
        print(click_lat_lng)
        return mru.picked_location(
            tfc.get_fcst(config), [""], click_lat_lng, latitude, longitude
        )


    @APP.callback(
        Output("local_graph", "figure"),
        Input("loc_marker", "position"),
        Input("start_date", "value"),
        Input("lead_time", "value"),
        Input("lead_time", "options"),
    )
    def local_plots(marker_pos, start_date, lead_time, targets):
        lat = marker_pos[0]
        lng = marker_pos[1]
        fcst_ds = tfc.get_fcst(config, start_date=start_date, lead_time=lead_time)
        # Errors handling
        error_msg = None
        if fcst_ds["prob"] is None:
            error_msg = "Data missing for this issue and target"
        else:
            try:
                fcst_ds = pingrid.sel_snap(fcst_ds, lat, lng)
                isnan = np.isnan(fcst_ds["prob"]).all()
                if isnan:
                    error_msg = "Data missing at this location"
            except KeyError:
                error_msg = "Grid box out of data domain"
        if error_msg is not None:
            local_graph = pingrid.error_fig(error_msg)
        else:
            for option in targets :
                if option["value"] == int(lead_time) :
                    target = option["label"]
            lng_units = "˚E" if (fcst_ds['X'] >= 0) else "˚W"
            lat_units = "˚N" if (fcst_ds['Y'] >= 0) else "˚S"
            local_graph = px.bar(
                (
                    fcst_ds["prob"]
                    .assign_coords(Terciles=("cat", ["Below", "Normal", "Above"]))
                    .to_dataframe()
                ),
                x="Terciles", y="prob", range_y=[0, 100], color="Terciles",
                ccolor_discrete_sequence=["yellow", "grey", "green"],
                title=(
                    f"{target} Forecast issued {start_date} at"
                    f" ({abs(fcst_ds['Y'].values)}{lat_units}"
                    f", {abs(fcst_ds['X'].values)}{lng_units})"
                )
            ).add_hline(
                y=100/3, line_dash="dash", line_color="black",
                annotation_text="climatology", annotation_position="bottom right",
            )
            y_ticks = [0, 37.5, 42.5, 47.5, 57.5, 67.5, 100]
            local_graph.update_layout(
                yaxis={
                    "tickmode":"array", "tickvals": y_ticks,
                    "ticktext": [f"{yt} %" for yt in y_ticks]}
            )
        return local_graph


    @APP.callback(
        Output("layers_control", "children"),
        Input("start_date","value"),
        Input("lead_time","value")
    )
    def make_map(start_date, lead_time):
        url_str = f"{TILE_PFX}/{{z}}/{{x}}/{{y}}/{start_date}/{lead_time}"
        return mru.layers_controls(
            url_str, "terc_fcst", "Forecast",
            GLOBAL_CONFIG["datasets"]["shapes_adm"], GLOBAL_CONFIG,
        )

    @FLASK.route(
        f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>/<start_date>/<lead_time>",
        endpoint=f"{config['core_path']}"
    )
    def fcst_tiles(tz, tx, ty, start_date, lead_time):
        # Reading
        fcst_ds = tfc.get_fcst(config, start_date=start_date, lead_time=lead_time)
        dominant_cat = fcst_ds["prob"].idxmax(dim="cat")
        fcst_class = (
            (fcst_ds["prob"] > 37.5) * 1.
            + (fcst_ds["prob"] > 42.5) * 1.
            + (fcst_ds["prob"] > 47.5) * 1.
            + (fcst_ds["prob"] > 57.5) * 1.
            + (fcst_ds["prob"] > 67.5) * 1.
        )
        # Can I do this without for? 
        # I am afraid not because where would not know how to broadcast...?
        dominant_fcst_class = sum([
            xr.where(
                # Where class is dominant
                fcst_ds["category"].isel(cat=c) == dominant_cat,
                # Translates class by 5c except for climatology that must remain 0
                # to get: 
                # Below (1, 2, 3, 4, 5) 
                # Normal (6, 7, 8, 9, 10)
                # Above (11, 12, 13, 14, 15)
                xr.where(
                    fcst_class.isel(cat=c) != 0, fcst_class.isel(cat=c)+ 5 * c, 0
                ),
                # Elsewhere set to 0 so can squeeze the list with sum
                0,
            )
            for c in range(3)
        ])
        dominant_fcst_class = xr.apply_ufunc(
            # Return all Normal classes to one class: 6
            # Reclassify Above to (7, 8, 9, 10, 11)
            np.piecewise, dominant_fcst_class,
            [
                dominant_fcst_class.data < 6,
                dominant_fcst_class.data > 10,
            ],
            [lambda x : x, lambda x : x - 4, 6],
        )
        dominant_fcst_class.attrs["colormap"] = CMAPS["prcp_terciles"]
        dominant_fcst_class.attrs["scale_min"] = 0
        dominant_fcst_class.attrs["scale_max"] = 11
        return pingrid.tile(
            dominant_fcst_class.rename({"X": "lon", "Y": "lat"}), tx, ty, tz
        )
