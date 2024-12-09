from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_leaflet as dlf
from controls import Block

from globals_ import GLOBAL_CONFIG

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
                        sm=12, md=4,
                        style={
                            "background-color": "white", "border-style": "solid",
                            "border-color": LIGHT_GRAY,
                            "border-width": "0px 1px 0px 0px",
                        },
                    ),
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        map_layout(),
                                        width=12,
                                        style={"background-color": "white"},
                                    ),
                                ],
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
                            ),
                        ],
                        sm=12,
                        md=8,
                        style={"background-color": "white"},
                    ),
                ],
            ),
        ],
        fluid=True,
        style={"padding-left": "0px", "padding-right": "0px"},
    )


def help_layout(buttonname, id_name, message):
    return html.Div(
        [
            html.Label(
                f"{buttonname}:", id=id_name,
                style={"cursor": "pointer","font-size": "100%","padding-left":"3px"},
            ),
            dbc.Tooltip(f"{message}", target=id_name, className="tooltiptext"),
        ]
    )


def navbar_layout():
    return dbc.Navbar(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.NavbarBrand("CCA", className="ml-2")
                        ),
                    ],
                    align="center", style={"padding-left":"5px"},
                ),
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            html.Div(
                [
                    help_layout(
                        "Probability",
                        "probability_label",
                        "Custom forecast probability choices",
                    ),
                ],
                style={
                    "color": "white", "position": "relative", "width": "95px",
                    "display": "inline-block", "padding": "10px",
                    "vertical-align": "top",
                }
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        id="proba",
                        clearable=False,
                        options=[
                            dict(label="exceeding", value="exceeding"),
                            dict(label="non-exceeding", value="non-exceeding"),
                        ],
                        value="exceeding",
                    )
                ],
                style={
                    "position": "relative", "width": "150px",
                    "display": "inline-block", "padding": "10px",
                    "vertical-align": "top",
                }
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        className="var",
                        id="variable",
                        clearable=False,
                        options=[
                            dict(label="Percentile", value="Percentile"),
                            dict(label="Value", value="Value"),
                        ],
                        value="Percentile",
                    )
                ],
                style={
                    "position": "relative", "width": "200px",
                    "display": "inline-block", "padding": "10px",
                    "vertical-align": "top",
                }
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        id="percentile",
                        clearable=False,
                        options=[
                            dict(label="10", value=0.1),
                            dict(label="15", value=0.15),
                            dict(label="20", value=0.2),
                            dict(label="25", value=0.25),
                            dict(label="30", value=0.3),
                            dict(label="35", value=0.35),
                            dict(label="40", value=0.4),
                            dict(label="45", value=0.45),
                            dict(label="50", value=0.5),
                            dict(label="55", value=0.55),
                            dict(label="60", value=0.60),
                            dict(label="65", value=0.65),
                            dict(label="70", value=0.70),
                            dict(label="75", value=0.75),
                            dict(label="80", value=0.8),
                            dict(label="85", value=0.85),
                            dict(label="90", value=0.9),
                        ],
                        value=0.5,
                    ),
                    html.Div([" %-ile"], style={
                        "color": "white", "font-size": "100%", "padding-top":"5px",
                        "padding-left":"3px",
                    })
                ],
            ),
            html.Div(
                [
                    dbc.Input(
                        id="threshold",
                        type="number",
                        className="my-1",
                        debounce=True,
                        value=0,
                    ),
                ],
            ),
            html.Div(
                [
                    help_layout(
                        "Forecast Issued",
                        "start_date_title",
                        "Model start dates",
                    ),
                ],
                style={
                    "color": "white", "position": "relative", "width": "145px",
                    "display": "inline-block", "padding": "10px",
                    "vertical-align": "top",
                }
            ),
            dbc.Alert(
                "Something went wrong",
                color="danger",
                dismissable=True,
                is_open=False,
                id="map_warning",
                style={"margin-bottom": "8px"},
            ),
        ],
        sticky="top", color=IRI_GRAY, dark=True,
    )


def controls_layout():
    return dbc.Container(
        [
            html.H5(["Climate Change Analysis"]),
            html.P(
                """
                This Maproom displays seasonal projected change of key climate
                variables with respect to historical records.
                """
            ),
            dcc.Loading(html.P(id="map_description"), type="dot"),
            html.P(
                """
                Use the controls in the top banner to choose other variables, models,
                scenarios, seasons, projected years and reference to compare with.
                """
            ),
            html.P(
                """
                Click the map (or enter coordinates) to show historical seasonal time
                series for this variable of this model, followed by a plume of
                possible projected scenarios.
                """
            ),
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
        ],
        fluid=True, className="scrollable-panel",
        style={"padding-bottom": "1rem", "padding-top": "1rem"},
    )


def map_layout():
    return dbc.Container(
        [
            html.H5(
                id="map_title",
                style={
                    "text-align":"center", "border-width":"1px",
                    "border-style":"solid", "border-color":"grey",
                    "margin-top":"3px", "margin-bottom":"3px",
                },
            ),
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
                        position="bottomleft",
                        width=300,
                        height=10,
                        min=0,
                        max=1,
                        nTicks=11,
                        opacity=1,
                        tooltip=True,
                        className="p-1",
                        style={
                            "background": "white", "border-style": "inset",
                            "-moz-border-radius": "4px", "border-radius": "4px",
                            "border-color": "LightGrey",
                        },
                    ),
                ],
                id="map",
                center=None,
                zoom=GLOBAL_CONFIG["zoom"],
                style={"width": "100%", "height": "50vh"},
            ),
        ],
        fluid=True,
    )


def results_layout():
    return dbc.Tabs(
        [
            dbc.Tab(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Spinner(
                                    dcc.Graph(id="local_graph"),
                                )
                            ),
                        ]
                    )
                ],
                label="Local History and Projections",
            )
        ],
        className="mt-4",
    )
