import os
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash import dash_table
import dash_leaflet as dlf
import plotly.express as px
from controls import Block, Sentence, DateNoYear, Number, Select, PickPoint

import numpy as np
from pathlib import Path
import calc
import pingrid
import pandas as pd

from globals_ import GLOBAL_CONFIG

CONFIG = GLOBAL_CONFIG["maprooms"]["crop_suitability"]

ZARR_PATH = GLOBAL_CONFIG["datasets"]["daily"]["vars"]["precip"][1]
if ZARR_PATH is None:
    ZARR_PATH = GLOBAL_CONFIG["datasets"]["daily"]["vars"]["precip"][0]
DR_PATH = f'{GLOBAL_CONFIG["datasets"]["daily"]["zarr_path"]}{ZARR_PATH}'
RR_MRG_ZARR = Path(DR_PATH)

IRI_BLUE = "rgb(25,57,138)"
IRI_GRAY = "rgb(113,112,116)"
LIGHT_GRAY = "#eeeeee"


def app_layout():

    # Initialization
    rr_mrg = calc.read_zarr_data(RR_MRG_ZARR)
    center_of_the_map = [((rr_mrg["Y"][int(rr_mrg["Y"].size/2)].values)), ((rr_mrg["X"][int(rr_mrg["X"].size/2)].values))]
    lat_res = np.around((rr_mrg["Y"][1]-rr_mrg["Y"][0]).values, decimals=10)
    lat_min = np.around((rr_mrg["Y"][0]-lat_res/2).values, decimals=10)
    lat_max = np.around((rr_mrg["Y"][-1]+lat_res/2).values, decimals=10)
    lon_res = np.around((rr_mrg["X"][1]-rr_mrg["X"][0]).values, decimals=10)
    lon_min = np.around((rr_mrg["X"][0]-lon_res/2).values, decimals=10)
    lon_max = np.around((rr_mrg["X"][-1]+lon_res/2).values, decimals=10)
    lat_label = str(lat_min)+" to "+str(lat_max)+" by "+str(lat_res)+"˚"
    lon_label = str(lon_min)+" to "+str(lon_max)+" by "+str(lon_res)+"˚"
    year_max = str(rr_mrg["T"][-1].dt.year.values)

    return dbc.Container(
        [
            dcc.Location(id="location", refresh=True),
            navbar_layout(),
            dbc.Row(
                [
                    dbc.Col(
                        controls_layout(
                            lat_min, lat_max,
                            lon_min, lon_max,
                            lat_label, lon_label,
                            year_max,
                        ),
                        sm=12,
                        md=4,
                        style={
                            "background-color": "white",
                            "border-style": "solid",
                            "border-color": LIGHT_GRAY,
                            "border-width": "0px 1px 0px 0px",
                            "overflow":"scroll","height":"95vh",#column that holds text and controls
                        },
                    ),
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        map_layout(center_of_the_map, lon_min, lat_min, lon_max, lat_max),
                                        width=12,
                                        style={
                                            "background-color": "white",
                                        },
                                    ),
                                ],
                                style={"overflow":"scroll","height": "55%"}, #box the map is in
                                className="g-0",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        results_layout(),
                                        width=12,
                                        style={
                                            "background-color": "white",
                                            "min-height": "100px",
                                            "border-style": "solid",
                                            "border-color": LIGHT_GRAY,
                                            "border-width": "1px 0px 0px 0px",
                                        },
                                    ),
                                ],
                                style={"overflow":"scroll","height":"45%"}, #box the plots are in
                                className="g-0",
                            ),
                        ],style={"overflow":"scroll","height":"95vh"},#main column for map and results
                        sm=12,
                        md=8,
                    ),
                ],
                className="g-0",
            ),
        ],
        fluid=True,
        style={"padding-left": "1px", "padding-right": "1px","height":"100%"},
    )


def navbar_layout():
    return dbc.Navbar(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(
                                src=f"{GLOBAL_CONFIG['url_path_prefix']}/static/{GLOBAL_CONFIG['logo']}",
                                height="30px",
                            )
                        ),
                        dbc.Col(
                            dbc.NavbarBrand(
                                "Climate and Agriculture / " + CONFIG["crop_suit_title"],
                                className="ml-2",
                            )
                        ),
                    ],
                    align="center",
                    className="g-0",
                ),
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                id="navbar-collapse",
                navbar=True,
            ),
        ],
        sticky="top",
        color=IRI_GRAY,
        dark=True,
    )


def controls_layout(lat_min, lat_max, lon_min, lon_max, lat_label, lon_label, year_max):
    return dbc.Container(
        [
            html.Div(
                [
                    html.H5(CONFIG["crop_suit_title"]),
                    html.P(
                        [f"""
                        Maproom to explore crop climate suitability using a 
                        spatial overlay analysis. The output is an index from 
                        0-5 
                        where 5 meet all conditions and is the most suitable.
                        """,
                        html.Br(),
                        """Select from the layers dropdown to select which data 
                        to view as timeseries data and as a map layer. You can 
                        Select a custom point to view for the timeseries data, 
                        as well as a specific year to view on the map.""",
                        html.Br(),
                        """Select custom values to determine the crop suitability 
                        parameters passed into the analysis to determine the 
                        most suitable conditions for your inquiry.""",
                        ]
                    )
                ]+[
                    html.H5("Dataset Documentation"),
                    html.P(
                        f"""
                        Reconstructed gridded rainfall, minimum temperature, and maximum temperature from {GLOBAL_CONFIG["institution"]}.
                        The time series data were created by combining
                        quality-controlled station observations in 
                        {GLOBAL_CONFIG["institution"]}’s archive with satellite rainfall estimates.
                       Crop suitability values use the rainfall, mimumum and maximum temperature data as inputs to calculate a suitability index where a certain number of conditions of crop suitability are met. 
                        """
                    ),
                ]+[
                    html.P([html.H6(val["menu_label"]), html.P(val["description"])])
                    for key, val in CONFIG["map_text"].items()
                ],
                style={"position":"relative","height":"25%", "overflow":"scroll"},#box holding text
            ),
            html.H3("Controls Panel",style={"padding":".5rem"}),
            html.Div(
                [
                    Block("Variable",
                        Select(
                            "data_choice",
                            [key for key, val in CONFIG["map_text"].items()],
                            labels=[val["menu_label"] for key, val in CONFIG["map_text"].items()],
                        ),
                    ),
                    Block(
                        "Pick a point",
                        PickPoint(lat_min, lat_max, lat_label, lon_min, lon_max, lon_label),
                        width="w-auto",
                    ),
                    Block("Map for a Year and a Season:",
                        dbc.Row(
                            [
                                dbc.Col(
                                    Number(
                                        "target_year",
                                        year_max,
                                        min=1981,
                                        max=year_max,
                                        width="6em",
                                    ),
                                ),
                                dbc.Col(
                                    Select(
                                        "target_season",
                                        ["DJF", "MAM", "JJA", "SON"],
                                        labels=["Dec-Feb", "Mar-May", "Jun-Aug", "Sep-Nov"],
                                    ),
                                ),
                            ],
                            class_name="g-0",
                            justify="start",
                        ),
                        width="w-auto",
                    ),
                    Block(
                        "Optimum seasonal total rainfall",
                        Sentence(
                            "Total rainfall amount between",
                            Number("lower_wet_threshold", CONFIG["param_defaults"]["lower_wet_thresh"], min=0, max=99999, width="6em"),
                            "and",
                            Number("upper_wet_threshold", CONFIG["param_defaults"]["upper_wet_thresh"], min=0, max=99999, width="6em"),
                            "mm",
                        ),
                    ),
                    Block(
                        "Temperature tolerance",
                        Sentence(
                            "Temperature range between",
                            Number("minimum_temp", CONFIG["param_defaults"]["min_temp"], min=-99, max=999, width="6em"),
                            "and",
                            Number("maximum_temp", CONFIG["param_defaults"]["max_temp"], min=-99, max=99999, width="6em"),
                            "C",
                        ),
                    ),
                    Block(
                        "Optimal daily temperature amplitude",
                        Sentence(
                            "An average daily temperature amplitude of:",
                            Number("temp_range", CONFIG["param_defaults"]["temp_range"], min=0, max=99999, width="4em"),
                            "C",
                        ),
                    ),
#                    Block(
#                        "Season length",
#                        Sentence(
#                            "The minimum length of the season:",
#                            Number("season_length", CONFIG["param_defaults"]["season_length"], min=0, max=99999, html_size=3),
#                            "days",
#                        ),
#                    ),
                    Block(
                        "Wet days",
                        Sentence(
                            "The minimum number of wet days within a season:",
                            Number("min_wet_days", CONFIG["param_defaults"]["min_wet_days"], min=0, max=99999, width="5em"),
                            "days",
                        ),
                        Sentence(
                            "Where a wet day is defined as a day with rainfall more than:",
                            Number("wet_day_def", CONFIG["param_defaults"]["wet_day_def"], min=0, max=9999, width="5em"),
                            "mm",
                        ),
                    ),
                ],
                style={"position":"relative","height":"60%", "overflow":"scroll"},#box holding controls
            ),
            html.Div(
                [dbc.Button(id="submit_params", children='Submit Crop Suitability Conditions', style={"position":"fixed", "width":"31%"})],
            )
        ], #end container
        fluid=True,
        className="scrollable-panel p-3",
        style={"overflow":"scroll","height":"100%","padding-bottom": "1rem", "padding-top": "1rem"},
    )    #style for container that is returned #95vh

def map_layout(center_of_the_map, lon_min, lat_min, lon_max, lat_max):
    return dbc.Container(
        [
            dlf.Map(
                [
                    dlf.LayersControl(id="layers_control", position="topleft"),
                    dlf.LayerGroup(
                        #Dummy position overridden by pick_location initial callback
                        #position must be defined
                        [dlf.Marker(id="loc_marker", position=[0, 0])],
                        id="layers_group"
                    ),
                    dlf.ScaleControl(imperial=False, position="topright"),
                    dlf.Colorbar(
                        id="colorbar",
                        position="bottomleft",
                        width=300,
                        height=10,
                        opacity=1,
                        className="p-1",
                        style={
                            "background": "white",
                            "border-style": "inset",
                            "-moz-border-radius": "4px",
                            "border-radius": "4px",
                            "border-color": "LightGrey",
                        },
                    )
                ],
                id="map",
                center=center_of_the_map,
                zoom=GLOBAL_CONFIG["zoom"],
                maxBounds = [[lat_min, lon_min],[lat_max, lon_max]],
                minZoom = GLOBAL_CONFIG["zoom"] - 1,
                maxZoom = GLOBAL_CONFIG["zoom"] + 10, #this was completely arbitrary
                style={
                    "width": "100%",
                    "height": "87%",#height of the map 
                },
            ),
            html.H6(
                id="map_title"
            ),
            html.H6(
                id="hover_feature_label"
            )
        ],
        fluid=True,
        style={"padding": "0rem", "height":"100%"},#box that holds map and title
    )


def results_layout():
    return html.Div( 
        [
            dbc.Spinner(dcc.Graph(id="timeseries_graph")),
        ],
        id="results_div",
    )
