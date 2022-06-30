import os
import flask
import dash
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import dash_leaflet as dlf
from pathlib import Path
import pyaconf
import pingrid
import layout
import plotly.graph_objects as pgo
import plotly.express as px
import numpy as np
import cptio
import xarray as xr
from scipy.stats import t, norm, rankdata
import pandas as pd

CONFIG = pyaconf.load(os.environ["CONFIG"])

PFX = CONFIG["core_path"]
TILE_PFX = CONFIG["tile_path"]
ADMIN_PFX = CONFIG["admin_path"]
DATA_PATH = CONFIG["results_path"]

# App

CONFIG = pyaconf.load(os.environ["CONFIG"])

SERVER = flask.Flask(__name__)
APP = dash.Dash(
    __name__,
    server=SERVER,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://use.fontawesome.com/releases/v5.12.1/css/all.css",
    ],
    url_base_pathname=f"{PFX}/",
    meta_tags=[
        {"name": "description", "content": "Seasonal Forecast"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)
APP.title = "Sub-Seasonal Forecast"

APP.layout = layout.app_layout()


@APP.callback(
    Output("map", "center"),
    Output("latInput", "min"),
    Output("latInput", "max"),
    Output("lngInput", "min"),
    Output("lngInput", "max"),
    Output("latLab", "children"),
    Output("lonLab", "children"),
    Input("submitLatLng","n_clicks"),
)
def initialize(toto):
    fcst_mu = cptio.open_cptdataset(Path(DATA_PATH, Path(CONFIG["forecast_mu_file"])))
    center_of_the_map = [((fcst_mu.Y[0]+fcst_mu.Y[-1])/2).values, ((fcst_mu.X[0]+fcst_mu.X[-1])/2).values]
    lat_res = np.around((fcst_mu.Y[0]-fcst_mu.Y[1]).values, decimals=10)
    lat_min = str(np.around((fcst_mu.Y[-1]-lat_res/2).values, decimals=10))
    lat_max = str(np.around((fcst_mu.Y[0]+lat_res/2).values, decimals=10))
    lon_res = np.around((fcst_mu.X[1]-fcst_mu.X[0]).values, decimals=10)
    lon_min = str(np.around((fcst_mu.X[0]-lon_res/2).values, decimals=10))
    lon_max = str(np.around((fcst_mu.X[-1]+lon_res/2).values, decimals=10))
    lat_label = lat_min+" to "+lat_max+" by "+str(lat_res)+"˚"
    lon_label = lon_min+" to "+lon_max+" by "+str(lon_res)+"˚"
    return center_of_the_map, lat_min, lat_max, lon_min, lon_max, lat_label, lon_label


def get_coords(click_lat_lng):
    if click_lat_lng is not None:
        return click_lat_lng
    else:
        fcst_mu = cptio.open_cptdataset(Path(DATA_PATH, Path(CONFIG["forecast_mu_file"])))
        return [(fcst_mu.Y[0].values+fcst_mu.Y[-1].values)/2, (fcst_mu.X[0].values+fcst_mu.X[-1].values)/2]


def round_latLng(coord):
    value = float(coord)
    value = round(value, 1)
    return value


@APP.callback(Output("map", "click_lat_lng"), Input("submitLatLng","n_clicks"), State("latInput", "value"), State("lngInput", "value"))
def inputCoords(n_clicks,latitude,longitude):
    if latitude is None:
        return None
    else:
        lat_lng = [latitude, longitude]
        return lat_lng


@APP.callback(Output("layers_group", "children"), Input("map", "click_lat_lng"))
def map_click(click_lat_lng):
    lat_lng = get_coords(click_lat_lng)

    return dlf.Marker(
        position=lat_lng, children=dlf.Tooltip("({:.3f}, {:.3f})".format(*lat_lng))
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
    Output("phys_units", "children"),
    Input("submitLatLng","n_clicks"),
)
def read_units(toto):

    fcst_mu = cptio.open_cptdataset(Path(DATA_PATH, Path(CONFIG["forecast_mu_file"])))
    fcst_mu_name = list(fcst_mu.data_vars)[0]
    return [" "+fcst_mu[fcst_mu_name].attrs["units"]]


@APP.callback(
    Output("cdf_graph", "figure"),
    Output("pdf_graph", "figure"),
    Input("map", "click_lat_lng"),
)
def local_plots(click_lat_lng):

    lat, lng = get_coords(click_lat_lng)

    # Reading
    fcst_mu = cptio.open_cptdataset(Path(DATA_PATH, Path(CONFIG["forecast_mu_file"])))
    fcst_mu_name = list(fcst_mu.data_vars)[0]
    fcst_mu = fcst_mu[fcst_mu_name]
    fcst_var = cptio.open_cptdataset(Path(DATA_PATH, Path(CONFIG["forecast_var_file"])))
    fcst_var_name = list(fcst_var.data_vars)[0]
    fcst_var = fcst_var[fcst_var_name]
    obs = cptio.open_cptdataset(Path(DATA_PATH, Path(CONFIG["obs_file"])))
    obs_name = list(obs.data_vars)[0]
    obs = obs[obs_name]
    if CONFIG["y_transform"]:
        hcst = cptio.open_cptdataset(Path(DATA_PATH, Path(CONFIG["hcst_file"])))
        hcst_name = list(hcst.data_vars)[0]
        hcst = hcst[hcst_name]
    # Spatial Tolerance for lat/lon selection clicking on map
    half_res = (fcst_mu["X"][1] - fcst_mu["X"][0]) / 2
    tol = np.sqrt(2 * np.square(half_res)) 
    
    # Errors handling
    try:
        isnan = (np.isnan(fcst_mu.sel(
            X=lng, Y=lat, method="nearest", tolerance=tol.values
        )).sum()) + (np.isnan(obs.sel(
            X=lng, Y=lat, method="nearest", tolerance=tol.values
        )).sum())
        if CONFIG["y_transform"]:
            isnan_yt = (np.isnan(hcst.sel(
                X=lng, Y=lat, method="nearest", tolerance=tol.values
            )).sum())
            isnan = isnan + isnan_yt
        if isnan > 0:
            errorFig = pgo.Figure().add_annotation(
                x=2,
                y=2,
                text="Data missing at this location",
                font=dict(family="sans serif", size=30, color="crimson"),
                showarrow=False,
                yshift=10,
                xshift=60,
            )
            return errorFig, errorFig
    except KeyError:
        errorFig = pgo.Figure().add_annotation(
            x=2,
            y=2,
            text="Grid box out of data domain",
            font=dict(family="sans serif", size=30, color="crimson"),
            showarrow=False,
            yshift=10,
            xshift=60,
        )
        return errorFig, errorFig
    
    fcst_mu = fcst_mu.sel(X=lng, Y=lat, method="nearest", tolerance=tol.values)
    fcst_var = fcst_var.sel(X=lng, Y=lat, method="nearest", tolerance=tol.values)
    obs = obs.sel(X=lng, Y=lat, method="nearest", tolerance=tol.values)
    
    # Get Issue date and Target season
    # Hard coded for now as I am not sure how we are going to deal with time
    issue_date = pd.to_datetime(["2022-04-01"]).strftime("%-d %b %Y").values[0]
    target_start = pd.to_datetime(["2022-04-02"]).strftime("%-d %b %Y").values[0]
    target_end = pd.to_datetime(["2022-04-08"]).strftime("%-d %b %Y").values[0]

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
    fcst_dof = int(fcst_var.attrs["dof"])
    if CONFIG["y_transform"]:
        hcst = hcst.sel(X=lng, Y=lat, method="nearest", tolerance=tol.values)
        hcst_err_var = (np.square(obs - hcst).sum(dim="T")) / fcst_dof
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

    # Graph for CDF
    cdf_graph = pgo.Figure()
    cdf_graph.add_trace(
        pgo.Scatter(
            x=fcst_ppf.squeeze().values,
            y=fcst_ppf["percentile"] * -1 + 1,
            hovertemplate="%{y:.0%} chance of exceeding"
            + "<br>%{x:.1f} "
            + fcst_mu.attrs["units"],
            name="forecast",
            line=pgo.scatter.Line(color="red"),
        )
    )
    cdf_graph.add_trace(
        pgo.Scatter(
            x=obs_ppf.values,
            y=fcst_ppf["percentile"] * -1 + 1,
            hovertemplate="%{y:.0%} chance of exceeding"
            + "<br>%{x:.1f} "
            + fcst_mu.attrs["units"],
            name="obs (parametric)",
            line=pgo.scatter.Line(color="blue"),
        )
    )
    cdf_graph.add_trace(
        pgo.Scatter(
            x=obs_quant.values,
            y=fcst_ppf["percentile"] * -1 + 1,
            hovertemplate="%{y:.0%} chance of exceeding"
            + "<br>%{x:.1f} "
            + fcst_mu.attrs["units"],
            name="obs (empirical)",
            line=pgo.scatter.Line(color="blue"),
        )
    )
    cdf_graph.update_traces(mode="lines", connectgaps=False)
    cdf_graph.update_layout(
        xaxis_title=fcst_mu_name + " " + "(" + fcst_mu.attrs["units"] + ")",
        yaxis_title="Probability of exceeding",
        title={
            "text": f"{target_start} - {target_end} forecast issued {issue_date} at ({round_latLng(lat)}N,{round_latLng(lng)}E)",
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
            customdata=fcst_ppf["percentile"] * -1 + 1,
            hovertemplate="%{customdata:.0%} chance of exceeding"
            + "<br>%{x:.1f} "
            + fcst_mu.attrs["units"],
            name="forecast",
            line=pgo.scatter.Line(color="red"),
        )
    )
    pdf_graph.add_trace(
        pgo.Scatter(
            x=obs_ppf.values,
            y=obs_pdf.values,
            customdata=fcst_ppf["percentile"] * -1 + 1,
            hovertemplate="%{customdata:.0%} chance of exceeding"
            + "<br>%{x:.1f} "
            + fcst_mu.attrs["units"],
            name="obs",
            line=pgo.scatter.Line(color="blue"),
        )
    )
    pdf_graph.update_traces(mode="lines", connectgaps=False)
    pdf_graph.update_layout(
        xaxis_title=fcst_mu_name + " " + "(" + fcst_mu.attrs["units"] + ")",
        yaxis_title="Probability density",
        title={
            "text": f"{target_start} - {target_end} forecast issued {issue_date} at ({round_latLng(lat)}N,{round_latLng(lng)}E)",
            "font": dict(size=14),
        },
    )
    return cdf_graph, pdf_graph


@APP.callback(
    Output("fcst_colorbar", "colorscale"),
    Input("proba", "value"),
    Input("variable", "value"),
    Input("percentile", "value")
)
def draw_colorbar(proba, variable, percentile): 

    fcst_cdf = xr.DataArray()
    if variable == "Percentile":
        if proba == "exceeding":
            percentile = 1 - percentile
            fcst_cdf.attrs["colormap"] = pingrid.RAIN_POE_COLORMAP
        else:
            fcst_cdf.attrs["colormap"] = pingrid.RAIN_PNE_COLORMAP
        thresholds = np.array([
            0,
            (percentile-0.05)/3,
            2*(percentile-0.05)/3,
            percentile-0.05,
            percentile-0.05+1/256.,
            percentile+0.05-1/256.,
            percentile+0.05,
            percentile+0.05+(1-(percentile+0.05))/3,
            percentile+0.05+2*(1-(percentile+0.05))/3,
            1
        ])
    else:
        fcst_cdf.attrs["colormap"] = pingrid.CORRELATION_COLORMAP
        thresholds = np.array([0, 0.1, 0.2, 0.35, 0.45, 0.45+1/256., 0.55-1/256., 0.55, 0.7, 0.85, 1])
    fcst_cs = pingrid.to_dash_colorscale(fcst_cdf.attrs["colormap"], thresholds=thresholds)
    return fcst_cs


@APP.callback(
    Output("fcst_layer", "url"),
    Output("forecast_warning", "is_open"),
    Input("proba", "value"),
    Input("variable", "value"),
    Input("percentile", "value"),
    Input("threshold", "value")
)
def fcst_tile_url_callback(proba, variable, percentile, threshold):

    try:
        if variable != "Percentile":
            if threshold is None:
                return "", True
            else:
                return f"{TILE_PFX}/{{z}}/{{x}}/{{y}}/{proba}/{variable}/{percentile}/{float(threshold)}", False
        else:
            return f"{TILE_PFX}/{{z}}/{{x}}/{{y}}/{proba}/{variable}/{percentile}/0.0", False
    except:
        return "", True


# Endpoints


@SERVER.route(
    f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>/<proba>/<variable>/<float:percentile>/<float:threshold>"
)
def fcst_tiles(tz, tx, ty, proba, variable, percentile, threshold):

    # Reading
    
    fcst_mu = cptio.open_cptdataset(Path(DATA_PATH, Path(CONFIG["forecast_mu_file"])))
    fcst_mu_name = list(fcst_mu.data_vars)[0]
    fcst_mu = fcst_mu[fcst_mu_name]
    fcst_var = cptio.open_cptdataset(Path(DATA_PATH, Path(CONFIG["forecast_var_file"])))
    fcst_var_name = list(fcst_var.data_vars)[0]
    fcst_var = fcst_var[fcst_var_name]
    obs = cptio.open_cptdataset(Path(DATA_PATH, Path(CONFIG["obs_file"])))
    obs_name = list(obs.data_vars)[0]
    obs = obs[obs_name]
    if CONFIG["y_transform"]:
        hcst = cptio.open_cptdataset(Path(DATA_PATH, Path(CONFIG["hcst_file"])))
        hcst_name = list(hcst.data_vars)[0]
        hcst = hcst[hcst_name]
    
    # Get Issue date and Target season
    # Hard coded for now as I am not sure how we are going to deal with time
    issue_date = pd.to_datetime(["2022-04-01"])
    target_start = pd.to_datetime(["2022-04-02"])
    target_end = pd.to_datetime(["2022-04-08"])

    # Obs CDF

    if variable == "Percentile":
        obs_mu = obs.mean(dim="T")
        obs_stddev = obs.std(dim="T")
        obs_ppf = xr.apply_ufunc(
            norm.ppf,
            percentile,
            kwargs={"loc": obs_mu, "scale": obs_stddev},
        )
    else:
        obs_ppf = threshold

    # Forecast CDF
    fcst_dof = int(fcst_var.attrs["dof"])
    if CONFIG["y_transform"]:
        hcst_err_var = (np.square(obs - hcst).sum(dim="T")) / fcst_dof
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
    ).squeeze("T")
    # Depending on choices:
    # probabilities symmetry around 0.5
    # choice of colorscale (dry to wet, wet to dry, or correlation)
    # translation of "near normal" to
    if variable == "Percentile":
        if proba == "exceeding":
            fcst_cdf = 1 - fcst_cdf
            percentile = 1 - percentile
            fcst_cdf.attrs["colormap"] = pingrid.RAIN_POE_COLORMAP
        else:
            fcst_cdf.attrs["colormap"] = pingrid.RAIN_PNE_COLORMAP
        fcst_cdf.attrs["colormapkey"] = np.array([
            0,
            (percentile-0.05)/3,
            2*(percentile-0.05)/3,
            percentile-0.05,
            percentile-0.05+1/256.,
            percentile+0.05-1/256.,
            percentile+0.05,
            percentile+0.05+(1-(percentile+0.05))/3,
            percentile+0.05+2*(1-(percentile+0.05))/3,
            1
        ])
    else:
        if proba == "exceeding":
            fcst_cdf = 1 - fcst_cdf
        fcst_cdf.attrs["colormap"] = pingrid.CORRELATION_COLORMAP
        fcst_cdf.attrs["colormapkey"] = np.array([0, 0.1, 0.2, 0.35, 0.45, 0.45+1/256., 0.55-1/256., 0.55, 0.7, 0.85, 1])
    fcst_cdf.attrs["scale_min"] = 0
    fcst_cdf.attrs["scale_max"] = 1
    clipping = None
    resp = pingrid.tile(fcst_cdf, tx, ty, tz, clipping)
    return resp


if __name__ == "__main__":
    import warnings
    warnings.simplefilter('error')
    APP.run_server(CONFIG["server"], CONFIG["port"], debug=CONFIG["mode"] != "prod")
