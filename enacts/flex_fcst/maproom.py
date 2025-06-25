import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import pingrid
from pingrid import CMAPS
from . import layout
import plotly.graph_objects as pgo
import numpy as np
import xarray as xr
from scipy.stats import t, norm
import pandas as pd
from . import predictions
from . import pycpt
import maproom_utilities as mapr_u
import dash_leaflet as dlf
from globals_ import GLOBAL_CONFIG
import calc

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
            Output("start_date", "options"),
            Output("start_date", "value"),
            Output("lead_time_label", "style"),
            Output("lead_time_control", "style"),
            Output("lat_input", "min"),
            Output("lat_input", "max"),
            Output("lat_input_tooltip", "children"),
            Output("lng_input", "min"),
            Output("lng_input", "max"),
            Output("lng_input_tooltip", "children"),
            Output("map", "center"),
            State("lead_time_label", "style"),
            State("lead_time_control", "style"),
            Input("location", "pathname"),
    )
    def initialize(lead_time_label_style, lead_time_control_style, path):
        #Initialization for start date dropdown to get a list of start dates
        # according to files available
        fcst_ds, obs = pycpt.get_fcst(GLOBAL_CONFIG["datasets"]["fcst_data"])
        start_dates = fcst_ds["S"].dt.strftime("%b-%-d-%Y").values
        if "Li" in fcst_ds.dims :
            lead_times = fcst_ds["Li"].values
            target_display = "inline-block" if (lead_times.size > 1) else "none"
        else :
            lead_times = None
            target_display = "none"
        lead_time_label_style = dict(lead_time_label_style, display=target_display)
        lead_time_control_style = dict(
            lead_time_control_style, display=target_display
        )
        center_of_the_map = [
            ((fcst_ds["Y"][int(fcst_ds["Y"].size/2)].values)),
            ((fcst_ds["X"][int(fcst_ds["X"].size/2)].values)),
            ]
        lat_res = (fcst_ds["Y"][0]-fcst_ds["Y"][1]).values
        lat_min = str((fcst_ds["Y"][-1]-lat_res/2).values)
        lat_max = str((fcst_ds["Y"][0]+lat_res/2).values)
        lon_res = (fcst_ds["X"][1]-fcst_ds["X"][0]).values
        lon_min = str((fcst_ds["X"][0]-lon_res/2).values)
        lon_max = str((fcst_ds["X"][-1]+lon_res/2).values)
        lat_label = lat_min+" to "+lat_max+" by "+str(lat_res)+"˚"
        lon_label = lon_min+" to "+lon_max+" by "+str(lon_res)+"˚"
        phys_units = [" "+obs.attrs["units"]] if "units" in obs.attrs else ""
        return (
            phys_units,
            start_dates, start_dates[-1],
            lead_time_label_style, lead_time_control_style,
            lat_min, lat_max, lat_label,
            lon_min, lon_max, lon_label,
            center_of_the_map
        )

    @APP.callback(
        Output("percentile_style", "style"),
        Output("threshold_style", "style"),
        Input("variable", "value")
    )
    def display_relevant_control(variable):

        displayed_style={
            "position": "relative",
            "width": "190px",
            "display": "flex",
            "padding": "10px",
            "vertical-align": "top",
        }
        if variable == "Percentile":
            style_percentile=displayed_style
            style_threshold={"display": "none"}
        else:
            style_percentile={"display": "none"}
            style_threshold=displayed_style
        return style_percentile, style_threshold


    @APP.callback(
        Output("lead_time","options"),
        Output("lead_time","value"),
        Input("start_date","value"),
    )
    def target_range_options(start_date):
        fcst_ds, _ = pycpt.get_fcst(GLOBAL_CONFIG["datasets"]["fcst_data"])
        return pycpt.pycptv2_targets_dict(fcst_ds, start_date=start_date)


    @APP.callback(
        Output("map_title", "children"),
        Input("start_date", "value"),
        Input("lead_time", "value"),
        Input("lead_time", "options"),
        Input("lead_time_control", "style"),
    )
    def write_map_title(start_date, lead_time, targets, lead_time_control):
        for option in targets :
            if option["value"] == lead_time :
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
        fcst_ds, _ = pycpt.get_fcst(GLOBAL_CONFIG["datasets"]["fcst_data"])
        if dash.ctx.triggered_id == None:
            lat = fcst_ds["Y"][int(fcst_ds["Y"].size/2)].values
            lng = fcst_ds["X"][int(fcst_ds["X"].size/2)].values
        else:
            if dash.ctx.triggered_id == "map":
                lat = click_lat_lng[0]
                lng = click_lat_lng[1]
            else:
                lat = latitude
                lng = longitude
            try:
                nearest_grid = pingrid.sel_snap(fcst_ds, lat, lng)
                lat = nearest_grid["Y"].values
                lng = nearest_grid["X"].values
            except KeyError:
                lat = lat
                lng = lng
        return [lat, lng], lat, lng


    @APP.callback(
        Output("cdf_graph", "figure"),
        Output("pdf_graph", "figure"),
        Input("loc_marker", "position"),
        Input("start_date", "value"),
        Input("lead_time", "value"),
        Input("lead_time", "options")
    )
    def local_plots(marker_pos, start_date, lead_time, targets):
        # Time Units Errors handling
        # Reading
        lat = marker_pos[0]
        lng = marker_pos[1]
        fcst_ds, obs = pycpt.get_fcst(GLOBAL_CONFIG["datasets"]["fcst_data"])
        fcst_ds = fcst_ds.sel(S=start_date)
        if "Li" in fcst_ds.dims :
            fcst_ds = fcst_ds.sel(Li=int(lead_time))
        obs = obs.where(
            obs["T"].dt.month == fcst_ds.squeeze()["T"].dt.month.values, drop=True,
        )
        # Errors handling
        try:
            if (
                fcst_ds["deterministic"] is None
                or fcst_ds["prediction_error_variance"] is None
                or obs is None
            ):
                error_fig = pingrid.error_fig(
                    error_msg="Data missing for this issue and target"
                )
                return error_fig, error_fig
            else:
                fcst_ds = pingrid.sel_snap(fcst_ds, lat, lng)
                obs = pingrid.sel_snap(obs, lat, lng)
                isnan = (
                    np.isnan(fcst_ds["deterministic"]).any()
                    or np.isnan(obs).any()
                )
                if config["y_transform"]:
                    if fcst_ds["hcst"] is None:
                        error_fig = pingrid.error_fig(
                            error_msg="Data missing for this issue and target"
                        )
                        return error_fig, error_fig
                    else:
                        isnan_yt = np.isnan(fcst_ds["hcst"]).any()
                        isnan = isnan or isnan_yt
                if isnan:
                    error_fig = pingrid.error_fig(
                        error_msg="Data missing at this location"
                    )
                    return error_fig, error_fig
        except KeyError:
            error_fig = pingrid.error_fig(error_msg="Grid box out of data domain")
            return error_fig, error_fig
        # CDF from 499 quantiles
        quantiles = np.arange(1, 500) / 500
        quantiles = xr.DataArray(
            quantiles, dims="percentile", coords={"percentile": quantiles}
        )
        # Obs CDF
        obs_q, obs_mu = xr.broadcast(quantiles, obs.mean(dim="T"))
        obs_stddev = obs.std(dim="T")
        obs_ppf = xr.apply_ufunc(
            norm.ppf,
            obs_q,
            kwargs={"loc": obs_mu, "scale": obs_stddev},
        ).rename("obs_ppf")
        # Obs quantiles
        obs_quant = obs.quantile(quantiles, dim="T")
        # Forecast CDF
        fcst_q, fcst_mu = xr.broadcast(quantiles, fcst_ds["deterministic"])
        try:
            fcst_dof = int(fcst_ds["fcst_var"].attrs["dof"])
        except:
            fcst_dof = obs["T"].size - 1
        if config["y_transform"]:
            hcst_err_var = (np.square(obs - fcst_ds["hcst"]).sum(dim="T")) / fcst_dof
            # fcst variance is hindcast variance weighted by (1+xvp)
            # but data files don't have xvp neither can we recompute it from them
            # thus xvp=0 is an approximation, acceptable dixit Simon Mason
            # The line below is thus just a reminder of the above
            xvp = 0
            fcst_var = hcst_err_var * (1 + xvp)
        else:
            fcst_var = fcst_ds["prediction_error_variance"]
        fcst_ppf = xr.apply_ufunc(
            t.ppf,
            fcst_q,
            fcst_dof,
            kwargs={
                "loc": fcst_mu,
                "scale": np.sqrt(fcst_var),
            },
        ).rename("fcst_ppf")
        if config["variable"] == "Precipitation":
            fcst_ppf = fcst_ppf.clip(min=0)
            obs_ppf = obs_ppf.clip(min=0)
        poe = fcst_ppf["percentile"] * -1 + 1
        # Graph for CDF
        cdf_graph = pgo.Figure()
        cdf_graph.add_trace(
            pgo.Scatter(
                x=fcst_ppf.squeeze().values,
                y=poe,
                hovertemplate="%{y:.0%} chance of exceeding"
                + "<br>%{x:.1f} "
                + obs.attrs["units"],
                name="forecast",
                line=pgo.scatter.Line(color="red"),
            )
        )
        cdf_graph.add_trace(
            pgo.Scatter(
                x=obs_ppf.values,
                y=poe,
                hovertemplate="%{y:.0%} chance of exceeding"
                + "<br>%{x:.1f} "
                + obs.attrs["units"],
                name="obs (parametric)",
                line=pgo.scatter.Line(color="blue"),
            )
        )
        cdf_graph.add_trace(
            pgo.Scatter(
                x=obs_quant.values,
                y=poe,
                hovertemplate="%{y:.0%} chance of exceeding"
                + "<br>%{x:.1f} "
                + obs.attrs["units"],
                name="obs (empirical)",
                line=pgo.scatter.Line(color="forestgreen"),
            )
        )
        cdf_graph.update_traces(mode="lines", connectgaps=False)
        for option in targets :
            if option["value"] == lead_time :
                target = option["label"]
        cdf_graph.update_layout(
            xaxis_title=f'{config["variable"]} ({obs.attrs["units"]})',
            yaxis_title="Probability of exceeding",
            title={
                "text": (
                    f'{target} forecast issued {start_date} '
                    f'<br> at ({fcst_ds["Y"].values}N,{fcst_ds["X"].values}E)'
                ),
                "font": dict(size=14),
            },
        )
        # PDF from 499 ppf values
        fcst_pdf = xr.apply_ufunc(
            t.pdf,
            fcst_ppf,
            fcst_dof,
            kwargs={
                "loc": fcst_ds["deterministic"],
                "scale": np.sqrt(fcst_var),
            },
        ).rename("fcst_pdf")
        obs_pdf = xr.apply_ufunc(
            norm.pdf,
            obs_ppf,
            kwargs={"loc": obs_mu, "scale": obs_stddev},
        ).rename("obs_pdf")
        # Graph for PDF
        pdf_graph = pgo.Figure()
        pdf_graph.add_trace(
            pgo.Scatter(
                x=fcst_ppf.squeeze().values,
                y=fcst_pdf.squeeze().values,
                customdata=poe,
                hovertemplate="%{customdata:.0%} chance of exceeding"
                + "<br>%{x:.1f} "
                + obs.attrs["units"],
                name="forecast",
                line=pgo.scatter.Line(color="red"),
            )
        )
        pdf_graph.add_trace(
            pgo.Scatter(
                x=obs_ppf.values,
                y=obs_pdf.values,
                customdata=poe,
                hovertemplate="%{customdata:.0%} chance of exceeding"
                + "<br>%{x:.1f} "
                + obs.attrs["units"],
                name="obs",
                line=pgo.scatter.Line(color="blue"),
            )
        )
        pdf_graph.update_traces(mode="lines", connectgaps=False)
        pdf_graph.update_layout(
            xaxis_title=f'{config["variable"]} ({obs.attrs["units"]})',
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
        Output("fcst_colorbar", "colorscale"),
        Input("proba", "value"),
        Input("variable", "value"),
        Input("percentile", "value")
    )
    def draw_colorbar(proba, variable, percentile):
        return to_flexible(
            xr.DataArray(), proba, variable, percentile,
        ).attrs["colormap"].to_dash_leaflet()


    @APP.callback(
        Output("layers_control", "children"),
        Output("forecast_warning", "is_open"),
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
                    url_str = f"{TILE_PFX}/{{z}}/{{x}}/{{y}}/{proba}/{variable}/{percentile}/{float(threshold)}/{start_date}/{lead_time}"
            else:
                send_alarm = False
                url_str = f"{TILE_PFX}/{{z}}/{{x}}/{{y}}/{proba}/{variable}/{percentile}/0.0/{start_date}/{lead_time}"
        except:
            url_str= ""
            send_alarm = True
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
                    url=url_str,
                    opacity=1,
                ),
                name="Forecast",
                checked=True,
            ),
        ], send_alarm


    # Endpoints

    @FLASK.route(
        f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>/<proba>/<variable>/<float:percentile>/<float(signed=True):threshold>/<start_date>/<lead_time>",
        endpoint=f"{config['core_path']}"
    )
    def fcst_tiles(tz, tx, ty, proba, variable, percentile, threshold, start_date, lead_time):
        # Reading
        fcst_ds, obs = pycpt.get_fcst(GLOBAL_CONFIG["datasets"]["fcst_data"])
        fcst_ds = fcst_ds.sel(S=start_date)
        if "Li" in fcst_ds.dims :
            fcst_ds = fcst_ds.sel(Li=int(lead_time))
        obs = obs.where(
            obs["T"].dt.month == fcst_ds.squeeze()["T"].dt.month.values, drop=True,
        )
        # Obs CDF
        if variable == "Percentile":
            obs_mu = obs.mean(dim="T")
            obs_stddev = obs.std(dim="T")
            obs_ppf = xr.apply_ufunc(
                norm.ppf,
                percentile,
                kwargs={"loc": obs_mu, "scale": obs_stddev},
            )
            if config["variable"] == "Precipitation":
                obs_ppf = obs_ppf.clip(min=0)
        else:
            obs_ppf = threshold
        # Forecast CDF
        try:
            fcst_dof = int(fcst_ds["fcst_var"].attrs["dof"])
        except:
            fcst_dof = obs["T"].size - 1
        if config["y_transform"]:
            hcst_err_var = (np.square(obs - fcst_ds["hcst"]).sum(dim="T")) / fcst_dof
            # fcst variance is hindcast variance weighted by (1+xvp)
            # but data files don't have xvp neither can we recompute it from them
            # thus xvp=0 is an approximation, acceptable dixit Simon Mason
            # The line below is thus just a reminder of the above
            xvp = 0
            fcst_var = hcst_err_var * (1 + xvp)
        else:
            fcst_var = fcst_ds["prediction_error_variance"]

        fcst_cdf = xr.DataArray( # pingrid.tile expects a xr.DA but obs_ppf is never that
            data = xr.apply_ufunc(
                t.cdf,
                obs_ppf,
                fcst_dof,
                kwargs={
                    "loc": fcst_ds["deterministic"],
                    "scale": np.sqrt(fcst_var),
                },
            ),
            # Naming conventions for pingrid.tile
            coords = fcst_ds.rename({"X": "lon", "Y": "lat"}).coords,
            dims = fcst_ds.rename({"X": "lon", "Y": "lat"}).dims
        # pingrid.tile wants 2D data
        ).squeeze()
        # Depending on choices:
        # probabilities symmetry around percentile threshold
        # choice of colorscale (dry to wet, wet to dry, or correlation)
        fcst_cdf = to_flexible(fcst_cdf, proba, variable, percentile,)
        clip_shape = calc.get_geom(level=0, conf=GLOBAL_CONFIG)["the_geom"][0]

        resp = pingrid.tile(fcst_cdf, tx, ty, tz, clip_shape)
        return resp
