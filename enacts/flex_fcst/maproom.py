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
from . import cpt
import urllib
import dash_leaflet as dlf
import psycopg2
from psycopg2 import sql
import shapely
from shapely import wkb
from shapely.geometry.multipolygon import MultiPolygon
from globals_ import FLASK, GLOBAL_CONFIG

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


    #Should I move this function into the predictions.py file where I put the other funcs?
    #if we do so maybe I should redo the func to be more flexible since it is hard coded to read each file separately..
    def read_cptdataset(lead_time, start_date, y_transform=config["y_transform"]):
        if config["leads"] is not None and config["targets"] is not None:
            raise Exception("I am not sure which of leads or targets to use")
        elif config["leads"] is not None:
            use_leads = lead_time
            use_targets = None
        elif config["targets"] is not None:
            use_leads = None
            use_targets = lead_time
        else:
            raise Exception("One of leads or targets must be not None")
        fcst_mu = cpt.read_file(
            config["forecast_path"],
            config["forecast_mu_file_pattern"],
            start_date,
            lead_time=use_leads,
            target_time=use_targets,
        )
        if fcst_mu is not None:
            fcst_mu_name = list(fcst_mu.data_vars)[0]
            fcst_mu = fcst_mu[fcst_mu_name]
        fcst_var = cpt.read_file(
            config["forecast_path"],
            config["forecast_var_file_pattern"],
            start_date,
            lead_time=use_leads,
            target_time=use_targets,
        )
        if fcst_var is not None:
            fcst_var_name = list(fcst_var.data_vars)[0]
            fcst_var = fcst_var[fcst_var_name]
        obs = cpt.read_file(
            config["forecast_path"],
            config["obs_file_pattern"],
            start_date,
            lead_time=use_leads,
            target_time=use_targets,
        )
        if obs is not None:
            obs = obs.squeeze()
            obs_name = list(obs.data_vars)[0]
            obs = obs[obs_name]
        if y_transform:
            hcst = cpt.read_file(
                config["forecast_path"],
                config["hcst_file_pattern"],
                start_date,
                lead_time=use_leads,
                target_time=use_targets,
            )
            if hcst is not None:
                hcst = hcst.squeeze()
                hcst_name = list(hcst.data_vars)[0]
                hcst = hcst[hcst_name]
        else:
            hcst = None
        return fcst_mu, fcst_var, obs, hcst

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
            Output("layers_control", "children"),
            State("lead_time_label", "style"),
            State("lead_time_control", "style"),
            Input("location", "pathname"),
    )
    def initialize(lead_time_label_style, lead_time_control_style, path):
        #Initialization for start date dropdown to get a list of start dates according to files available
        if config["forecast_mu_file_pattern"] is None:
            fcst_mu, fcst_var, obs = cpt.read_pycptv2dataset(config["forecast_path"])
            start_dates = fcst_mu["S"].dt.strftime("%b-%-d-%Y").values
        else:
            start_dates = cpt.starts_list(
                config["forecast_path"],
                config["forecast_mu_file_pattern"],
                config["start_regex"],
                format_in=config["start_format_in"],
                format_out=config["start_format_out"],
            )

        if config["forecast_mu_file_pattern"] is None:
            fcst_mu, fcst_var, obs = cpt.read_pycptv2dataset(config["forecast_path"])
        else:
            if config["leads"] is not None and config["targets"] is not None:
                raise Exception("I am not sure which of leads or targets to use")
            elif config["leads"] is not None:
                use_leads = list(config["leads"])[0]
                use_targets = None
            elif config["targets"] is not None:
                use_leads = None
                use_targets = config["targets"][-1]
            else:
                raise Exception("One of leads or targets must be not None")
            fcst_mu = cpt.read_file(
                config["forecast_path"],
                config["forecast_mu_file_pattern"],
                start_dates[-1],
                lead_time=use_leads,
                target_time=use_targets,
            )
        center_of_the_map = [((fcst_mu["Y"][int(fcst_mu["Y"].size/2)].values)), ((fcst_mu["X"][int(fcst_mu["X"].size/2)].values))]
        lat_res = (fcst_mu["Y"][0]-fcst_mu["Y"][1]).values
        lat_min = str((fcst_mu["Y"][-1]-lat_res/2).values)
        lat_max = str((fcst_mu["Y"][0]+lat_res/2).values)
        lon_res = (fcst_mu["X"][1]-fcst_mu["X"][0]).values
        lon_min = str((fcst_mu["X"][0]-lon_res/2).values)
        lon_max = str((fcst_mu["X"][-1]+lon_res/2).values)
        lat_label = lat_min+" to "+lat_max+" by "+str(lat_res)+"˚"
        lon_label = lon_min+" to "+lon_max+" by "+str(lon_res)+"˚"
        if config["forecast_mu_file_pattern"] is None:
            phys_units = [" "+obs.attrs["units"]]
            target_display = "inline-block" if "L" in fcst_mu.dims else "none"
        else:
            fcst_mu_name = list(fcst_mu.data_vars)[0]
            phys_units = [" "+fcst_mu[fcst_mu_name].attrs["units"]]
            target_display = "inline-block"
        lead_time_label_style = dict(lead_time_label_style, display=target_display)
        lead_time_control_style = dict(lead_time_control_style, display=target_display)

        layers = [
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
                    opacity=1,
                    id="tile_layer",
                ),
                name="Forecast",
                checked=True,
            ),
        ]
        return (
            phys_units,
            start_dates, start_dates[-1],
            lead_time_label_style, lead_time_control_style,
            lat_min, lat_max, lat_label,
            lon_min, lon_max, lon_label,
            center_of_the_map,
            layers,
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
        if config["forecast_mu_file_pattern"] is None:
            fcst_mu, fcst_var, obs = cpt.read_pycptv2dataset(config["forecast_path"])
            if "L" in fcst_mu.dims:
                fcst_mu = fcst_mu.sel(S=start_date)
                options = [
                    {
                        "label": predictions.target_range_formatting(
                            fcst_mu['Ti'].isel(S=0, L=ln, missing_dims="ignore").values,
                            fcst_mu['Tf'].isel(S=0, L=ln,  missing_dims="ignore").values,
                            "months",
                        ),
                        "value": lead,
                    } for ln, lead in enumerate(fcst_mu["L"].values)
                ]
                return options, options[0]["value"]
            else:
                return None, None
        else:
            if config["leads"] is not None and config["targets"] is not None:
                raise Exception("I am not sure which of leads or targets to use")
            elif config["leads"] is not None:
                leads_values = list(config["leads"].values())
                leads_keys = list(config["leads"])
                default_choice = list(config["leads"])[0]
            elif config["targets"] is not None:
                leads_values = config["targets"]
                leads_keys = leads_values
                default_choice = config["targets"][-1]
            else:
                raise Exception("One of leads or targets must be not None")
            start_date = pd.to_datetime(start_date)
            leads_dict = {}
            for idx, lead in enumerate(leads_keys):
                if config["leads"] is not None and config["targets"] is not None:
                    raise Exception("I am not sure which of leads or targets to use")
                elif config["leads"] is not None:
                    target_range = predictions.target_range_format(
                        leads_values[idx],
                        start_date,
                        config["target_period_length"],
                        config["time_units"],
                    )
                elif config["targets"] is not None:
                    target_range = leads_values[idx]
                else:
                    raise Exception("One of leads or targets must be not None")
                leads_dict.update({lead:target_range})
            return leads_dict, default_choice


    @APP.callback(
    Output("map_title","children"),
    Input("start_date","value"),
    Input("lead_time","value"),
    Input("lead_time","options"),
    )
    def write_map_title(start_date, lead_time, lead_time_options):
        if config["forecast_mu_file_pattern"] is None :
            fcst_mu, fcst_var, obs = cpt.read_pycptv2dataset(config["forecast_path"])
            if "L" not in fcst_mu.dims:
                fcst_mu = fcst_mu.sel(S=start_date)
                target_period = predictions.target_range_formatting(
                    fcst_mu['Ti'].isel(S=0, missing_dims="ignore").values,
                    fcst_mu['Tf'].isel(S=0, missing_dims="ignore").values,
                    "months"
                )
            else:
                for label, value in lead_time_options.items() :
                    if value == lead_time :
                        target_period = label
        else:
            for label, value in lead_time_options.items() :
                if value == lead_time :
                    target_period = label
        return f'{target_period} {config["variable"]} Forecast issued {start_date}'


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
        if config["forecast_mu_file_pattern"] is None:
            fcst_mu, fcst_var, obs = cpt.read_pycptv2dataset(config["forecast_path"])
            start_dates = fcst_mu["S"].dt.strftime("%b-%-d-%Y").values
        else:
            start_dates = cpt.starts_list(
                config["forecast_path"],
                config["forecast_mu_file_pattern"],
                config["start_regex"],
                format_in=config["start_format_in"],
                format_out=config["start_format_out"],
            )
            if config["leads"] is not None and config["targets"] is not None:
                raise Exception("I am not sure which of leads or targets to use")
            elif config["leads"] is not None:
                use_leads = list(config["leads"])[0]
                use_targets = None
            elif config["targets"] is not None:
                use_leads = None
                use_targets = config["targets"][-1]
            else:
                raise Exception("One of leads or targets must be not None")
            fcst_mu = cpt.read_file(
                config["forecast_path"],
                config["forecast_mu_file_pattern"],
                start_dates[-1],
                lead_time=use_leads,
                target_time=use_targets,
            )
        if dash.ctx.triggered_id == None:
            lat = fcst_mu["Y"][int(fcst_mu["Y"].size/2)].values
            lng = fcst_mu["X"][int(fcst_mu["X"].size/2)].values
        else:
            if dash.ctx.triggered_id == "map":
                lat = click_lat_lng[0]
                lng = click_lat_lng[1]
            else:
                lat = latitude
                lng = longitude
            try:
                nearest_grid = pingrid.sel_snap(fcst_mu, lat, lng)
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
        Input("start_date","value"),
        Input("lead_time","value"),
    )
    def local_plots(marker_pos, start_date, lead_time):
        # Time Units Errors handling
        if config["forecast_mu_file_pattern"] is None:
            start_date_pretty = (pd.to_datetime(start_date)).strftime("%b %Y")
        else:
            if config["time_units"] == "days":
                start_date_pretty = (pd.to_datetime(start_date)).strftime("%-d %b %Y")
            elif config["time_units"] == "months":
                start_date_pretty = (pd.to_datetime(start_date)).strftime("%b %Y")
            else:
                raise Exception("Forecast target time units should be days or months")
        # Reading
        lat = marker_pos[0]
        lng = marker_pos[1]
        if config["forecast_mu_file_pattern"] is None:
            fcst_mu, fcst_var, obs = cpt.read_pycptv2dataset(config["forecast_path"])
            fcst_mu = fcst_mu.sel(S=start_date)
            fcst_var = fcst_var.sel(S=start_date)
            if "L" in fcst_mu.dims:
                fcst_mu = fcst_mu.sel(L=lead_time)
                fcst_var = fcst_var.sel(L=lead_time)
            obs = obs.where(obs["T"].dt.month == fcst_mu["T"].dt.month, drop=True)
            is_y_transform = False
        else:
            fcst_mu, fcst_var, obs, hcst = read_cptdataset(lead_time, start_date, y_transform=config["y_transform"])
            is_y_transform = config["y_transform"]

        # Errors handling
        try:
            if fcst_mu is None or fcst_var is None or obs is None:
                error_fig = pingrid.error_fig(error_msg="Data missing for this issue and target")
                return error_fig, error_fig
            else:
                fcst_mu = pingrid.sel_snap(fcst_mu, lat, lng)
                fcst_var = pingrid.sel_snap(fcst_var, lat, lng)
                obs = pingrid.sel_snap(obs, lat, lng)
                isnan = np.isnan(fcst_mu) or np.isnan(obs).any()
                if is_y_transform:
                    if hcst is None:
                        error_fig = pingrid.error_fig(error_msg="Data missing for this issue and target")
                        return error_fig, error_fig
                    else:
                        hcst = pingrid.sel_snap(hcst, lat, lng)
                        isnan_yt = np.isnan(hcst).any()
                        isnan = isnan or isnan_yt
                if isnan:
                    error_fig = pingrid.error_fig(error_msg="Data missing at this location")
                    return error_fig, error_fig
        except KeyError:
            error_fig = pingrid.error_fig(error_msg="Grid box out of data domain")
            return error_fig, error_fig

        if config["forecast_mu_file_pattern"] is None:
            target_range = predictions.target_range_formatting(
                fcst_mu['Ti'].isel(S=0, L=0, missing_dims="ignore").values,
                fcst_mu['Tf'].isel(S=0, L=0, missing_dims="ignore").values,
                "months"
            )
        else:
            if config["leads"] is not None and config["targets"] is not None:
                raise Exception("I am not sure which of leads or targets to use")
            elif config["leads"] is not None:
                target_range = predictions.target_range_format(
                    config["leads"][lead_time],
                    pd.to_datetime(start_date),
                    config["target_period_length"],
                    config["time_units"],
                )
            elif config["targets"] is not None:
                target_range = lead_time
            else:
                raise Exception("One of leads or targets must be not None")
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
        fcst_q, fcst_mu = xr.broadcast(quantiles, fcst_mu)
        try:
            fcst_dof = int(fcst_var.attrs["dof"])
        except:
            fcst_dof = obs["T"].size - 1
        if config["y_transform"]:
            hcst_err_var = (np.square(obs - hcst).sum(dim="T")) / fcst_dof
            # fcst variance is hindcast variance weighted by (1+xvp)
            # but data files don't have xvp neither can we recompute it from them
            # thus xvp=0 is an approximation, acceptable dixit Simon Mason
            # The line below is thus just a reminder of the above
            xvp = 0
            fcst_var = hcst_err_var * (1 + xvp)
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
                line=pgo.scatter.Line(color="blue"),
            )
        )
        cdf_graph.update_traces(mode="lines", connectgaps=False)
        cdf_graph.update_layout(
            xaxis_title=f'{config["variable"]} ({obs.attrs["units"]})',
            yaxis_title="Probability of exceeding",
            title={
                "text": f'{target_range} forecast issued {start_date_pretty} <br> at ({fcst_mu["Y"].values}N,{fcst_mu["X"].values}E)',
                "font": dict(size=14),
            },
        )

        # PDF from 499 ppf values

        fcst_pdf = xr.apply_ufunc(
            t.pdf,
            fcst_ppf,
            fcst_dof,
            kwargs={
                "loc": fcst_mu,
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
                "text": f'{target_range} forecast issued {start_date_pretty} <br> at ({fcst_mu["Y"].values}N,{fcst_mu["X"].values}E)',
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
        Output("tile_layer", "url"),
        Output("forecast_warning", "is_open"),
        Input("proba", "value"),
        Input("variable", "value"),
        Input("percentile", "value"),
        Input("threshold", "value"),
        Input("start_date","value"),
        Input("lead_time","value")
    )
    def set_tile_url(proba, variable, percentile, threshold, start_date, lead_time):

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
        return url_str, send_alarm


    # Endpoints

    @FLASK.route(
        f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>/<proba>/<variable>/<float:percentile>/<float(signed=True):threshold>/<start_date>/<lead_time>",
        endpoint=f"{config['core_path']}"
    )
    def fcst_tiles(tz, tx, ty, proba, variable, percentile, threshold, start_date, lead_time):
        # Reading
        
        if config["forecast_mu_file_pattern"] is None:
            fcst_mu, fcst_var, obs = cpt.read_pycptv2dataset(config["forecast_path"])
            fcst_mu = fcst_mu.sel(S=start_date)
            fcst_var = fcst_var.sel(S=start_date)
            if "L" in fcst_mu.dims:
                fcst_mu = fcst_mu.sel(L=int(lead_time))
                fcst_var = fcst_var.sel(L=int(lead_time))
            obs = obs.where(obs["T"].dt.month == fcst_mu["T"].dt.month, drop=True)
            is_y_transform = False
        else:
            fcst_mu, fcst_var, obs, hcst = read_cptdataset(lead_time, start_date, y_transform=config["y_transform"])
            is_y_transform = config["y_transform"]
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
            fcst_dof = int(fcst_var.attrs["dof"])
        except:
            fcst_dof = obs["T"].size - 1
        if is_y_transform:
            hcst_err_var = (np.square(obs - hcst).sum(dim="T")) / fcst_dof
            # fcst variance is hindcast variance weighted by (1+xvp)
            # but data files don't have xvp neither can we recompute it from them
            # thus xvp=0 is an approximation, acceptable dixit Simon Mason
            # The line below is thus just a reminder of the above
            xvp = 0
            fcst_var = hcst_err_var * (1 + xvp)

        fcst_cdf = xr.DataArray( # pingrid.tile expects a xr.DA but obs_ppf is never that
            data = xr.apply_ufunc(
                t.cdf,
                obs_ppf,
                fcst_dof,
                kwargs={
                    "loc": fcst_mu,
                    "scale": np.sqrt(fcst_var),
                },
            ),
            # Naming conventions for pingrid.tile
            coords = fcst_mu.rename({"X": "lon", "Y": "lat"}).coords,
            dims = fcst_mu.rename({"X": "lon", "Y": "lat"}).dims
        # pingrid.tile wants 2D data
        ).squeeze()
        # Depending on choices:
        # probabilities symmetry around percentile threshold
        # choice of colorscale (dry to wet, wet to dry, or correlation)
        fcst_cdf = to_flexible(fcst_cdf, proba, variable, percentile,)
        with psycopg2.connect(**GLOBAL_CONFIG["db"]) as conn:
            s = sql.Composed([sql.SQL(GLOBAL_CONFIG['datasets']['shapes_adm'][0]['sql'])])
            df = pd.read_sql(s, conn)
            clip_shape = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))[0]

        resp = pingrid.tile(fcst_cdf, tx, ty, tz, clip_shape)
        return resp
