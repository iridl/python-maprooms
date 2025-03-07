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
import pandas as pd

from globals_ import GLOBAL_CONFIG

CONFIG = GLOBAL_CONFIG["maprooms"]["onset"]
IRI_BLUE = "rgb(25,57,138)"
IRI_GRAY = "rgb(113,112,116)"
LIGHT_GRAY = "#eeeeee"


def app_layout():

    return dbc.Container(
        [
            dcc.Location(id="location", refresh=True),
            navbar_layout(),
            dbc.Row(
                [
                    dbc.Col(
                        controls_layout(),
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
                                        map_layout(),
                                        width=12,
                                        style={
                                            "background-color": "white",
                                        },
                                    ),
                                ],
                                style={"overflow":"scroll","height": "45%"}, #box the map is in
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
                                style={"overflow":"scroll","height":"55%"}, #box the plots are in
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
                                id="navbarbrand",
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


def controls_layout():
    return dbc.Container(
        [
            html.Div(
                [
                    html.H5(id="app_title"),
                    html.P([
                        (
                            f"""
                            The Maproom explores current and historical rainy season onset
                            """
                        ),
                        html.Span(id="andcess"),
                        (
                            f"""
                            dates based on user-defined definitions.
                            The date when the rainy season starts with germinating rains
                            is critical to agriculture planification, in particular for planting.
                            """
                        ),
                    ]),
                    dcc.Loading(html.P(id="map_description"), type="dot"),
                    html.P([
                        (
                            f"""
                            The Control Panel below allows to make other maps
                            and change the definition of the onset
                            """
                        ),
                        html.Span(id="andcess2"),
                        f"""dates.""",
                    ]),
                    html.P([
                        (
                            f"""
                            The local information shows first whether
                            the germinating rains have occured or not and when.
                            Graphics of historical onset
                            """
                        ),
                        html.Span(id="andcess3"),
                        (
                            f"""
                            dates are presented in the form of time series
                            and probability of exceeding.
                            Pick another point with the controls below
                            or by clicking on the map.
                            """
                        ),
                    ]),
                    html.P([
                        (
                            f"""
                            By enabling the exploration of the current and historical onset
                            """
                        ),
                        html.Span(id="andcess4"),
                        (
                            f"""
                            dates, the Maproom allows to monitor
                            and understand the spatial and temporal variability of how seasons unfold and
                            therefore characterize the risk for a successful
                            agricultural campaign.
                            """
                        ),
                    ]),
                    html.P(
                        """
                        The definition of the onset can be set up in the Controls above
                        and is looking at a significantly wet event (e.g. 20mm in 3 days),
                        called the germinating rains, that is not followed by a dry spell
                        (e.g. 7-day dry spell in the following 21 days).
                        The actual date is the first wet day of the wet event.
                        The onset date is computed on-the-fly for each year according to the definition,
                        and is expressed in days since an early start date
                        (e.g. Jun. 1st). The search for the onset date is made
                        from that early start date and for a certain number of
                        following days (e.g. 60 days). The early start date
                        serves as a reference and should be picked so that it
                        is ahead of the expected onset date.
                        """
                    ),
                    html.Div(id="maplabdesc"),
                    html.P([
                        f"""Note that if the criteria to define the onset""",
                        html.Span(id="andcess5"),
                        (
                            f"""
                            date are not met within the search period, the analysis will
                            return a missing value. And if the analysis returns 0,
                            it is likely that the onset"""
                        ),
                        html.Span(id="andcess6"),
                        (
                            f"""
                            has already occured and thus that the date from when to
                            search is within"""
                        ),
                        html.Span(id="andcess7"),
                        f"""the rainy season.""",
                    ]),
                    html.H5("Dataset Documentation"),
                    html.P(
                        f"""
                        Reconstructed gridded rainfall from {GLOBAL_CONFIG["institution"]}.
                        The time series were created by combining
                        quality-controlled station observations in 
                        {GLOBAL_CONFIG["institution"]}’s archive with satellite rainfall estimates.
                        """
                    ),
                ],
                style={"position":"relative","height":"30%", "overflow":"scroll"},#box holding text
            ),
            html.H3("Controls Panel",style={"padding":".5rem"}),
            html.Div(
                [
                    Block("Pick a point",
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.FormFloating([dbc.Input(
                                        id="lat_input", type="number",
                                    ),
                                    dbc.Label("Latitude", style={"font-size": "80%"}),
                                    dbc.Tooltip(
                                        id="lat_input_tooltip",
                                        target="lat_input",
                                        className="tooltiptext",
                                    )]),
                                ),
                                dbc.Col(
                                    dbc.FormFloating([dbc.Input(
                                        id="lng_input", type="number",
                                    ),
                                    dbc.Label("Longitude", style={"font-size": "80%"}),
                                    dbc.Tooltip(
                                        id="lng_input_tooltip",
                                        target="lng_input",
                                        className="tooltiptext",
                                    )]),
                                ),
                                dbc.Button(id="submit_lat_lng", children='Submit'),
                            ],
                        ),
                    ),
                    Block("Ask the map:",
                        html.Div(id="mapchoice"),
                        html.P(
                            Sentence(
                                Number("prob_exc_thresh_onset", 30, min=0, width="5em"),
                                html.Span(id="pet_units1"),
                                "?"
                            ),
                            id="pet_input_wrap_onset"
                        ),
                        html.P(
                            Sentence(
                                Number("prob_exc_thresh_length", 90, min=0, width="5em"),
                                "days?",
                            ),
                            id="pet_input_wrap_length"
                        ),
                        html.P(
                            Sentence(
                                Number("prob_exc_thresh_tot", 200, min=0, width="6em"),
                                "mm?"
                            ),
                            id="pet_input_wrap_tot"
                        ),
                    ),
                    Block(
                        "Onset Date Search Period",
                        html.Div(id="onsetsearchperiod"),
                    ),
                    Block(
                        "Wet Day Definition",
                        Sentence(
                            "Rainfall amount greater than",
                            Number("wet_threshold", 1, min=0, max=99999, width="5em"),
                            "mm",
                        ),
                    ),
                    Block(
                        "Onset Date Definition",
                        html.Div(id="onsetdef"),
                    ),
                    html.Div(id="cessdef"),
                ],
                style={"position":"relative","height":"60%", "overflow":"scroll"},#box holding controls
            ),
        ], #end container
        fluid=True,
        className="scrollable-panel p-3",
        style={"overflow":"scroll","height":"100%","padding-bottom": "1rem", "padding-top": "1rem"},
    )    #style for container that is returned #95vh

def map_layout():
    return dbc.Container(
        [
            dlf.Map(
                [
                    dlf.LayersControl(id="layers_control", position="topleft"),
                    dlf.LayerGroup(
                        [dlf.Marker(id="loc_marker", position=(0, 0))],
                        id="layers_group"
                    ),
                    dlf.ScaleControl(imperial=False, position="bottomleft"),
                    dlf.Colorbar(
                        id="colorbar",
                        min=0,
                        position="topright",
                        width=10,
                        height=200,
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
                center=None,
                zoom=GLOBAL_CONFIG["zoom"],
                minZoom = GLOBAL_CONFIG["zoom"] - 1,
                maxZoom = GLOBAL_CONFIG["zoom"] + 10, #this was completely arbitrary
                style={
                    "width": "100%",
                    "height": "77%",#height of the map 
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
            dbc.Tabs(
                [
                    dbc.Tab(
                        [
                            html.H6(id="germination_sentence"),
                            dbc.Spinner(dcc.Graph(id="onset_date_plot")),
                            dbc.Spinner(dcc.Graph(id="onset_prob_exc")),
                        ],
                        label="Onset Date",
                    ),
                    dbc.Tab(
                        [
                            dbc.Spinner(dcc.Graph(id="cess_date_plot")),
                            dbc.Spinner(dcc.Graph(id="cess_prob_exc")),
                        ],
                        id="cess_tab",
                        label="Cessation Date",
                    ),
                    dbc.Tab(
                        [
                            dbc.Spinner(dcc.Graph(id="length_plot")),
                            dbc.Spinner(dcc.Graph(id="length_prob_exc")),
                        ],
                        id="length_tab",
                        label="Season Length",
                    ),
                ],
                className="mt-4",
            )
        ],
    )
