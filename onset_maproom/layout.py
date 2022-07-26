import os
import pyaconf
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash import dash_table
import dash_leaflet as dlf
import plotly.express as px
from widgets import Block, Sentence, Date, Units, Number

import numpy as np
from pathlib import Path
import calc


CONFIG = pyaconf.load(os.environ["CONFIG"])
DR_PATH = CONFIG["rr_mrg_zarr_path"]
RR_MRG_ZARR = Path(DR_PATH)

IRI_BLUE = "rgb(25,57,138)"
IRI_GRAY = "rgb(113,112,116)"
LIGHT_GRAY = "#eeeeee"


def app_layout():

    # Initialization
    rr_mrg = calc.read_zarr_data(RR_MRG_ZARR)
    center_of_the_map = [((rr_mrg.Y[0]+rr_mrg.Y[-1])/2).values, ((rr_mrg.X[0]+rr_mrg.X[-1])/2).values]
    lat_res = np.around((rr_mrg.Y[1]-rr_mrg.Y[0]).values, decimals=10)
    lat_min = str(np.around((rr_mrg.Y[0]-lat_res/2).values, decimals=10))
    lat_max = str(np.around((rr_mrg.Y[-1]+lat_res/2).values, decimals=10))
    lon_res = np.around((rr_mrg.X[1]-rr_mrg.X[0]).values, decimals=10)
    lon_min = str(np.around((rr_mrg.X[0]-lon_res/2).values, decimals=10))
    lon_max = str(np.around((rr_mrg.X[-1]+lon_res/2).values, decimals=10))
    lat_label = lat_min+" to "+lat_max+" by "+str(lat_res)+"˚"
    lon_label = lon_min+" to "+lon_max+" by "+str(lon_res)+"˚"

    return dbc.Container(
        [
            dcc.Location(id="location", refresh=True),
            navbar_layout(),
            dbc.Row(
                [
                    dbc.Col(
                        controls_layout(lat_min, lat_max, lon_min, lon_max, lat_label, lon_label),
                        sm=12,
                        md=4,
                        style={
                            "background-color": "white",
                            "border-style": "solid",
                            "border-color": LIGHT_GRAY,
                            "border-width": "0px 1px 0px 0px",
                        },
                    ),
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        map_layout(center_of_the_map),
                                        width=12,
                                        style={
                                            "background-color": "white",
                                        },
                                    ),
                                ],
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
                                style={"overflow":"scroll","height":"50vh"},
                                className="g-0",
                            ),
                        ],
                        sm=12,
                        md=8,
                        style={"background-color": "white"},
                    ),
                ],
                className="g-0",
            ),
        ],
        fluid=True,
        style={"padding-left": "0px", "padding-right": "0px"},
    )


def navbar_layout():
    return dbc.Navbar(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(
                                src="assets/" + CONFIG["logo"],
                                height="30px",
                            )
                        ),
                        dbc.Col(
                            dbc.NavbarBrand(
                                "Climate and Agriculture / " + CONFIG["onset_and_cessation_title"],
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


def controls_layout(lat_min, lat_max, lon_min, lon_max, lat_label, lon_label):
    return dbc.Container(
        [
            html.Div(
                [
                    html.H5(
                        [
                            CONFIG["onset_and_cessation_title"],
                        ]
                    ),
                    html.P(
                        f"""
                        The Maproom explores current and historical rainy season onset
                        {" and cessation" if CONFIG["ison_cess_date_hist"] else "" }
                         dates based on user-defined definitions.
                        The date when the rainy season starts with germinating rains
                        is critical to agriculture planification, in particular for planting.
                        """
                    ),
                    html.P(
                        """
                        The default map shows whether germinating rains have occured
                        as of now (or the most recent rainfall data), and if so: when.
                        Dates are expressed in days since an Early Start date
                        configurable via the controls below
                        (see more details on onset date definition and other map controls below).
                        """
                    ),
                    html.P(
                        f"""
                        The default local information shows first whether
                        the germinating rains have occured or not and when.
                        Graphics of historical onset
                        {" and cessation" if CONFIG["ison_cess_date_hist"] else "" }
                        dates are presented in the form of time series
                        and probability of exceeding.
                        Pick another point with the controls below
                        or by clicking on the map.
                        """
                    ),
                    html.P(
                        f"""
                        By enabling the exploration of the current and historical onset
                        {" and cessation" if CONFIG["ison_cess_date_hist"] else "" }
                         dates, the Maproom allows to monitor
                        and understand the spatial and temporal variability of how seasons unfold and
                        therefore characterize the risk for a successful
                        agricultural campaign.
                        """
                    ),
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
                    html.H6("""Has germinating rain occured?"""),
                    html.P(
                        """
                        Shows the result of the germinating rain date search
                        (i.e. near-real time thus without checking of follow-up dry spells)
                        from the most recent Early Start to now (or the last day with available rainfall data).
                        """
                    ),
                    html.H6("""When to prepare for planting?"""),
                    html.P(
                        """
                        Shows the average onset date over all years of available data.
                        """
                    ),
                    html.H6("""How risky to plant..."""),
                    html.P(
                        """
                        Shows the probability of the onset date to be past a certain date,
                        through all the years of available data.
                        """
                    ),
                    html.P(
                        """
                        Note that if the criteria to define the onset date are
                        not met within the search period, the analysis will
                        return a missing value. And if the analysis returns 0
                        (days since the early start), it is likely that the
                        onset has already occured and thus that the
                        early start date picked is within the rainy season.
                        """
                    ),
                    html.H5("Dataset Documentation"),
                    html.P(
                        f"""
                        Reconstructed gridded rainfall from {CONFIG["institution"]}.
                        The time series were created by combining
                        quality-controlled station observations in 
                        {CONFIG["institution"]}’s archive with satellite rainfall estimates.
                        """
                    ),
                ],
                style={"position":"relative","height":"30vh", "overflow":"scroll"}, #,"height":"20%"
            ),
            html.H3("Controls Panel",style={"padding":"1rem 1rem"}),
            html.Div(
                [
                    Block("Pick a point",
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.FormFloating([dbc.Input(
                                        id = "latInput",
                                        min=lat_min,
                                        max=lat_max,
                                        type="number",
                                    ),
                                    dbc.Label(lat_label, style={"font-size": "80%"})]),
                                ),
                                dbc.Col(
                                    dbc.FormFloating([dbc.Input(
                                        id = "lngInput",
                                        min=lon_min,
                                        max=lon_max,
                                        type="number",
                                    ),
                                    dbc.Label(lon_label, style={"font-size": "80%"})]),
                                ),
                                dbc.Button(id="submitLatLng", n_clicks=0, children='Submit'),
                            ],
                        ),
                    ),
                    #Block(
                    #    "Date",
                    #    dbc.Select(
                    #        id="date_input",
                    #        value="onset",
                    #        bs_size="sm",
                    #        options=[
                    #            {"label": "Onset", "value": "onset"},
                    #            {"label": "Cessation", "value": "cessation"},
                    #        ],
                    #    ),
                    #),
                    Block(
                        "Ask the map:",
                        dbc.Select(
                            id="map_choice",
                            value="monit",
                            # bs_size="sm",
                            options=[
                                {"label": "Has germinating rain occured?", "value": "monit"},
                                {"label": "When to prepare for planting?", "value": "mean"},
                                {"label": "Climatological Standard deviation", "value": "stddev"},
                                {"label": "How risky to plant...", "value": "pe"},
                            ],
                        ),
                        html.P(
                            Sentence(
                                Number("probExcThresh1", 30, min=0),
                                html.Span(id="pet_units"),
                                "?"
                            ),
                            id="pet_input_wrapper"
                        )
                    ),
                    Block(
                        "Onset Date Search Period",
                        Sentence(
                            "From Early Start date of",
                            Date("search_start_", 1, CONFIG["default_search_month"]),
                            "and within the next",
                            Number("searchDays", 90, min=0, max=9999), "days",
                        ),
                    ),
                    Block(
                        "Wet Day Definition",
                        Sentence(
                            "Rainfall amount greater than",
                            Number("wetThreshold", 1, min=0, max=99999),
                            "mm",
                        ),
                    ),
                    Block(
                        "Onset Date Definition",
                        Sentence(
                            "First spell of",
                            Number("runningDays", CONFIG["default_running_days"], min=0, max=999),
                            "days that totals",
                            Number("runningTotal", 20, min=0, max=99999),
                            "mm or more and with at least",
                            Number("minRainyDays", CONFIG["default_min_rainy_days"], min=0, max=999),
                            "wet day(s) that is not followed by a",
                            Number("dryDays", 7, min=0, max=999),
                            "-day dry spell within the next",
                            Number("drySpell", 21, min=0, max=9999),
                            "days",
                        ),
                    ),
                    Block(
                        "Cessation Date Definition",
                        Sentence(
                            "First date after",
                            Date("start_cess_", 1, "Sep"),
                            "in",
                            Number("searchDaysCess", 90, min=0, max=99999),
                            "days when the soil water balance falls below",
                            Number("waterBalanceCess", 5, min=0, max=999),
                            "mm for a period of",
                            Number("drySpellCess", 3, min=0, max=999),
                            "days",
                        ),
                        ison=CONFIG["ison_cess_date_hist"]
                    ),
                ],
                style={"position":"relative","height":"70vh", "overflow":"scroll"},
            ),
        ], #end container
        fluid=True,
        className="scrollable-panel p-3",
        style={"overflow":"scroll","height":"100vh","padding-bottom": "1rem", "padding-top": "1rem"},
    )

def map_layout(center_of_the_map):
    return dbc.Container(
        [
            dlf.Map(
                [
                    dlf.LayersControl(
                        [
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
                            dlf.Overlay(
                                dlf.GeoJSON(
                                    id="borders_adm1",
                                    data={"features": []},
                                    options={
                                        "fill": False,
                                        "color": "black",
                                        "weight": 4,
                                    },
                                ),
                                name=CONFIG["name_adm1"],
                                checked=True,
                            ),
                            dlf.Overlay(
                                dlf.GeoJSON(
                                    id="borders_adm2",
                                    data={"features": []},
                                    options={
                                        "fill": False,
                                        "color": "grey",
                                        "weight": 3,
                                        "opacity": 0.8
                                    },
                                ),
                                name=CONFIG["name_adm2"],
                                checked=True,
                            ),
                            dlf.Overlay(
                                dlf.GeoJSON(
                                    id="borders_adm3",
                                    data={"features": []},
                                    options={
                                        "fill": False,
                                        "color": "grey",
                                        "weight": 2,
                                        "opacity": 0.6
                                    },
                                ),
                                name=CONFIG["name_adm3"],
                                checked=False,
                            ),
                            dlf.Overlay(
                                dlf.TileLayer(
                                    opacity=1,
                                    id="onset_layer",
                                ),
                                name="Onset",
                                checked=True,
                            ),
                        ],
                        position="topleft",
                        id="layers_control",
                    ),
                    dlf.LayerGroup(id="layers_group"),
                    dlf.ScaleControl(imperial=False, position="topright"),
                    dlf.Colorbar(
                        id="colorbar",
                        min=0,
                        position="bottomleft",
                        width=300,
                        height=10,
                        opacity=.8,
                    ),
                ],
                id="map",
                center=center_of_the_map,
                zoom=CONFIG["zoom"],
                style={
                    "width": "100%",
                    "height": "45vh",
                },
            ),
            html.H6(
                id="map_title"
            )
        ],
        fluid=True,
        style={"padding": "1rem"},
    )


def results_layout():
    return html.Div( 
        [   
            dbc.Tabs(
                [
                    dbc.Tab(
                        [
                            html.H6(id="germination_sentence"),
                            dbc.Spinner(dcc.Graph(id="onsetDate_plot")),
                            dbc.Spinner(dcc.Graph(id="probExceed_onset")),
                        ],
                        label="Onset Date",
                    ),
                    dbc.Tab(
                        [
                            dbc.Spinner(dcc.Graph(id="cessDate_plot")),
                            dbc.Spinner(dcc.Graph(id="probExceed_cess")),
                        ],
                        id="cess_dbct",
                        label="Cessation Date",
                    ),
                ],
                className="mt-4",
            )
        
            # return html.Img(
            #     style={"width": "600px"},
            #     src=(
            #         "https://iridl.ldeo.columbia.edu/dlcharts/render/"
            #         "905cdac6e87a58586967e115a18e615d01530ddd?_wd=1200px&_ht=600px"
            #         "&_langs=en&_mimetype=image%2Fpng"
            #         "&region=bb%3A39.375%3A7.125%3A39.625%3A7.375%3Abb"
            #         "&waterBalanceCess=3&drySpellCess=10&plotrange2=15"
            #     ),
            # )
        ], #end div
        style={"height":"60vh"}
    )
