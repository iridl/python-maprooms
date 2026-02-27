import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import pingrid
from pingrid import CMAPS
from . import layout
import plotly.graph_objects as pgo
import numpy as np
import xarray as xr
import flex_fcst_calc as ffc
import maproom_utilities as mru
from globals_ import GLOBAL_CONFIG
from fieldsets import Select


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
    APP.title = "Forecast"

    APP.layout = layout.app_layout()

    @APP.callback(
            Output("phys-units" ,"children"),
            Output("start_div", "children"),
            Output("lead_div", "children"),
            Output("target_block", "style"),
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
        #Initialization for start date dropdown to get a list of start dates
        # according to files available
        fcst_ds, obs, start_dates, target_display = ffc.get_fcst(config)
        phys_units = [" "+obs.attrs["units"]] if "units" in obs.attrs else ""
        issue_select = Select("start_date", options=start_dates)
        target_dict = ffc.get_targets_dict(config, fcst_ds, start_dates[0])
        lead_select = Select("lead_time",
            options=[ld["value"] for ld in target_dict],
            labels=[ld["label"] for ld in target_dict],
        )
        lead_style = {"display": "block"} if target_display else {"display": "none"}
        return (
            phys_units, issue_select, lead_select, lead_style,
        ) + mru.initialize_map(fcst_ds)

    @APP.callback(
        Output("perc_block", "style"),
        Output("thresh_block", "style"),
        Input("variable", "value")
    )
    def display_relevant_control(variable):
        if variable == "Percentile":
            style_percentile = {"display": "block"}
            style_threshold = {"display": "none"}
        else:
            style_percentile = {"display": "none"}
            style_threshold = {"display": "block"}
        return style_percentile, style_threshold


    @APP.callback(
        Output("lead_time","options"),
        Input("start_date","value"),
    )
    def target_range_options(start_date):
        fcst_ds, _, _, _ = ffc.get_fcst(config)
        return ffc.get_targets_dict(config, fcst_ds, start_date)


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
        # Reading
        fcst_ds, _, _, _ = ffc.get_fcst(config)
        return mru.picked_location(
            fcst_ds, [""], click_lat_lng, latitude, longitude
        )


    @APP.callback(
        Output("local_graph_0", "figure"),
        Output("local_graph_1", "figure"),
        Input("loc_marker", "position"),
        Input("start_date", "value"),
        Input("lead_time", "value"),
        Input("lead_time", "options"),
    )
    def local_plots(marker_pos, start_date, lead_time, targets):
        # Time Units Errors handling
        # Reading
        lat = marker_pos[0]
        lng = marker_pos[1]
        fcst_ds, obs, _, _ = ffc.get_fcst(config, start_date=start_date, lead_time=lead_time)
        # Errors handling
        fcst_ds, obs, error_msg = ffc.errors_handling(config, fcst_ds, obs, lat, lng)
        if error_msg is not None:
            error_fig = pingrid.error_fig(error_msg)
            return error_fig, error_fig
        else: 
            # CDF from 499 quantiles
            quantiles = np.arange(1, 500) / 500
            quantiles = xr.DataArray(
                quantiles, dims="percentile", coords={"percentile": quantiles}
            )
            (
                cdf_fcst_x, cdf_fcst_y, cdf_obs_x, cdf_obs_y,
                cdf_obs_emp_x, cdf_obs_emp_y,
                pdf_fcst_x, pdf_fcst_y, pdf_obs_x, pdf_obs_y,
            ) = ffc.distribution(quantiles, obs, fcst_ds, config)
            phys_units = (" "+obs.attrs["units"]) if "units" in obs.attrs else ""
            # Graph for CDF
            cdf_graph = pgo.Figure()
            cdf_graph.add_trace(
                pgo.Scatter(
                    x=cdf_fcst_x.squeeze().values,
                    y=cdf_fcst_y,
                    hovertemplate="%{y:.0%} chance of exceeding"
                    + "<br>%{x:.1f} "
                    + phys_units,
                    name="forecast",
                    line=pgo.scatter.Line(color="red"),
                )
            )
            cdf_graph.add_trace(
                pgo.Scatter(
                    x=cdf_obs_x.values,
                    y=cdf_obs_y,
                    hovertemplate="%{y:.0%} chance of exceeding"
                    + "<br>%{x:.1f} "
                    + phys_units,
                    name="obs (parametric)",
                    line=pgo.scatter.Line(color="blue"),
                )
            )
            if config["method"] == "pycpt" :
                cdf_graph.add_trace(
                    pgo.Scatter(
                        x=cdf_obs_emp_x.values,
                        y=cdf_obs_emp_y,
                        hovertemplate="%{y:.0%} chance of exceeding"
                        + "<br>%{x:.1f} "
                        + phys_units,
                        name="obs (empirical)",
                        line=pgo.scatter.Line(color="forestgreen"),
                    )
                )
            cdf_graph.update_traces(mode="lines", connectgaps=False)
            for option in targets :
                if option["value"] == int(lead_time) :
                    target = option["label"]
            cdf_graph.update_layout(
                xaxis_title=f'{config["variable"]} ({phys_units})',
                yaxis_title="Probability of exceeding",
                title={
                    "text": (
                        f'{target} forecast issued {start_date} '
                        f'<br> at ({fcst_ds["Y"].values}N,{fcst_ds["X"].values}E)'
                    ),
                    "font": dict(size=14),
                },
            )
            # Graph for PDF
            pdf_graph = pgo.Figure()
            pdf_graph.add_trace(
                pgo.Scatter(
                    x=pdf_fcst_x.squeeze().values,
                    y=pdf_fcst_y.squeeze().values,
                    customdata=(quantiles["percentile"] * -1 + 1),
                    hovertemplate="%{customdata:.0%} chance of exceeding"
                    + "<br>%{x:.1f} "
                    + phys_units,
                    name="forecast",
                    line=pgo.scatter.Line(color="red"),
                )
            )
            pdf_graph.add_trace(
                pgo.Scatter(
                    x=pdf_obs_x.values,
                    y=pdf_obs_y.values,
                    customdata=(quantiles["percentile"] * -1 + 1),
                    hovertemplate="%{customdata:.0%} chance of exceeding"
                    + "<br>%{x:.1f} "
                    + phys_units,
                    name="obs",
                    line=pgo.scatter.Line(color="blue"),
                )
            )
            pdf_graph.update_traces(mode="lines", connectgaps=False)
            pdf_graph.update_layout(
                xaxis_title=f'{config["variable"]} ({phys_units})',
                yaxis_title="Probability density",
                title={
                    "text": (
                        f'{target} forecast issued {start_date} '
                        f'<br> at ({fcst_ds["Y"].values}N,{fcst_ds["X"].values}E)'
                    ),
                    "font": dict(size=14),
                },
            )
            return cdf_graph, pdf_graph


    def to_flexible(fcst_cdf, proba, variable, percentile):
        if variable == "Percentile":
            if proba == "exceeding":
                fcst_cdf = 1 - fcst_cdf
                percentile = 1 - percentile
                color_scale = CMAPS["rain_poe"]
            else:
                color_scale = CMAPS["rain_pne"]
            scale = [
                0,
                (percentile - 0.05) * 1/3,
                (percentile - 0.05) * 2/3,
                percentile - 0.05,
                percentile - 0.05,
                percentile + 0.05,
                percentile + 0.05,
                percentile + 0.05 + (1 - (percentile + 0.05)) * 1/3,
                percentile + 0.05 + (1 - (percentile + 0.05)) * 2/3,
                1,
            ]
            color_scale = pingrid.ColorScale(
                color_scale.name, color_scale.colors, scale=scale,
            )
        else:
            if proba == "exceeding":
                fcst_cdf = 1 - fcst_cdf
            color_scale = CMAPS["correlation"]
        fcst_cdf.attrs["colormap"] = color_scale
        fcst_cdf.attrs["scale_min"] = 0
        fcst_cdf.attrs["scale_max"] = 1
        return fcst_cdf


    @APP.callback(
        Output("colorbar", "colorscale"),
        Input("proba", "value"),
        Input("variable", "value"),
        Input("percentile", "value")
    )
    def draw_colorbar(proba, variable, percentile):
        return to_flexible(
            xr.DataArray(), proba, variable, int(percentile)/100.,
        ).attrs["colormap"].to_dash_leaflet()


    @APP.callback(
        Output("layers_control", "children"),
        Input("proba", "value"),
        Input("variable", "value"),
        Input("percentile", "value"),
        Input("threshold", "value"),
        Input("start_date","value"),
        Input("lead_time","value")
    )
    def make_map(proba, variable, percentile, threshold, start_date, lead_time):
        try:
            if variable != "Percentile":
                if threshold is None:
                    url_str = ""
                    send_alarm = True
                else:
                    send_alarm = False
                    url_str = (
                        f"{TILE_PFX}/{{z}}/{{x}}/{{y}}/{proba}/{variable}/"
                        f"{percentile}/{float(threshold)}/{start_date}/{lead_time}"
                    )
            else:
                send_alarm = False
                url_str = (
                    f"{TILE_PFX}/{{z}}/{{x}}/{{y}}/{proba}/{variable}/"
                    f"{percentile}/0.0/{start_date}/{lead_time}"
                )
        except:
            url_str= ""
        return mru.layers_controls(
            url_str, "flex_fcst", "Forecast",
            GLOBAL_CONFIG["datasets"]["shapes_adm"], GLOBAL_CONFIG,
        )


    # Endpoints

    @FLASK.route(
        (
            f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>/<proba>/<variable>/"
            f"<percentile>/<float(signed=True):threshold>/<start_date>/<lead_time>"
        ),
        endpoint=f"{config['core_path']}"
    )
    def fcst_tiles(tz, tx, ty, proba, variable, percentile, threshold, start_date, lead_time):
        # Reading
        fcst_ds, obs, _, _ = ffc.get_fcst(config, start_date=start_date, lead_time=lead_time)
        fcst_cdf = ffc.proba(variable, obs, int(percentile)/100., config, threshold, fcst_ds)
        # Depending on choices:
        # probabilities symmetry around percentile threshold
        # choice of colorscale (dry to wet, wet to dry, or correlation)
        fcst_cdf = to_flexible(
            fcst_cdf, proba, variable, int(percentile)/100.
        )
        resp = pingrid.tile(fcst_cdf, tx, ty, tz)
        return resp
