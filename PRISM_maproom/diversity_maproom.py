import geopandas as gpd
from geopandas import GeoDataFrame
import os
import flask
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import dash_leaflet as dlf
from pathlib import Path
import pyaconf
import layout
import plotly.graph_objects as pgo
import plotly.express as px
import pandas as pd
import numpy as np
import json

DATA_path = "/data/drewr/PRISM/eBird/derived/detectionProbability/Mass_towns/"
df = pd.read_csv("/data/drewr/PRISM/eBird/derived/detectionProbability/originalCSV/bhco_weekly_DP_MAtowns_05_18.csv")
with open(f"{DATA_path}ma_towns.json") as geofile:
    towns = json.load(geofile)

mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A" #I took this from Xandre's code, not sure if I should generate a new one

SERVER = flask.Flask(__name__)
APP = dash.Dash(
    __name__,
    server=SERVER,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://use.fontawesome.com/releases/v5.12.1/css/all.css",
    ],
    #url_base_pathname=f"{PFX}/",
    meta_tags=[
        {"name": "description", "content": "Onset Maproom"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)
APP.title = "PRISM Maproom"

APP.layout = layout.app_layout()

@APP.callback(
    Output("choropleth", "figure"),
    [Input("date_dropdown", "value"), Input("candidate", "value")])
def display_choropleth(date, candidate):
    dfLoc = df.loc[df['date'] == date]
    fig = px.choropleth_mapbox(dfLoc, geojson=towns, 
        featureidkey="properties.city", 
        color=candidate, 
        locations = "city", 
        mapbox_style="carto-positron", 
        opacity=1,
        center={"lat":42, "lon": -71.3824},
        zoom = 7
    )
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0}, 
        mapbox_accesstoken=mapbox_access_token,
        title= f"{candidate} data for {date}"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.show()
    return fig

@APP.callback(
    Output("timeSeriesPlot", "figure"),
    [Input("city_dropdown", "value")])
def update_timeSeries(city):
    dfCity = df.loc[df['city'] == city]
    timeSeries = px.line(
        data_frame = dfCity,
        x = "date",
        y = "eBird.DP.RF"
    )
    timeSeries.update_traces(
        mode="markers+lines",
        hovertemplate='%{y} %{x}',
        connectgaps=False       
    )
    timeSeries.update_layout(
        yaxis_title="Detection probability",
        xaxis_title="dates"
    )
    return timeSeries

if __name__ == "__main__":
    APP.run_server(debug=True)
