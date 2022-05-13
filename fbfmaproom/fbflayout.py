from dash import dcc
from dash import html
from dash import dash_table as table
import dash_leaflet as dlf
import dash_leaflet.express as dlx
import dash_bootstrap_components as dbc
from dataclasses import dataclass

def app_layout():
    return html.Div(
        [
            dcc.Location(id="location", refresh=True),
            map_layout(),
            logo_layout(),
            table_layout(),
            command_layout(),
            disclaimer_layout(),
        ]
    )


def help_layout(buttonname, id_name, message):
    return html.Div(
        [
            html.Label(f"{buttonname}:", id=id_name, style={"cursor": "pointer"}),
            dbc.Tooltip(
                f"{message}",
                target=id_name,
                className="tooltiptext",
            ),
        ]
    )


def map_layout():
    return dlf.Map(
        [
            dlf.LayersControl(
                [
                    dlf.BaseLayer(
                        dlf.TileLayer(
                            url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
                        ),
                        name="Street",
                        checked=True,
                    ),
                    dlf.BaseLayer(
                        dlf.TileLayer(
                            url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
                        ),
                        name="Topo",
                        checked=False,
                    ),
                    dlf.Overlay(
                        dlf.GeoJSON(
                            id="borders",
                            data={"features": []},
                            options={
                                "fill": False,
                                "color": "black",
                                "weight": .25,
                            },
                        ),
                        name="Borders",
                        checked=True,
                    ),
                    dlf.Overlay(
                        dlf.TileLayer(opacity=0.8, id="pnep_layer"),
                        name="Forecast",
                        checked=True,
                    ),
                    dlf.Overlay(
                        dlf.TileLayer(opacity=0.8, id="vuln_layer"),
                        name="Vulnerability",
                        checked=False,
                    ),
                ],
                position="topleft",
                id="layers_control",
            ),
            dlf.LayerGroup(
                [
                    dlf.Polygon(
                        positions=[(0, 0), (0, 0)],
                        color="rgb(49, 109, 150)",
                        fillColor="orange",
                        fillOpacity=0.1,
                        weight=2,
                        id="feature",
                    ),
                    dlf.Marker(
                        [
                            dlf.Popup(id="marker_popup"),
                        ],
                        position=(0, 0),
                        draggable=True,
                        id="marker",
                    ),
                ],
                id="pixel_layer",
            ),
            dlf.ScaleControl(imperial=False, position="topleft"),
            dlf.Colorbar(
                "Vulnerability",
                id="vuln_colorbar",
                position="bottomleft",
                width=300,
                height=10,
                min=0,
                max=5,
                nTicks=5,
                opacity=0.8,
            ),
            dlf.Colorbar(
                id="pnep_colorbar",
                position="bottomleft",
                width=300,
                height=10,
                min=0,
                max=100,
                nTicks=5,
                opacity=0.8,
                tooltip=True,
            ),
        ],
        id="map",
        style={
            "width": "100%",
            "height": "100%",
            "position": "absolute",
        },
    )


def logo_layout():
    return html.Div(
        [html.H4("FBF—Maproom"), html.Img(id="logo")],
        id="logo_panel",
        className="info",
        style={
            "position": "absolute",
            "top": "10px",
            "width": "120px",
            "left": "90px",
            "z-index": "1000",
            "height": "fit-content",
            "pointer-events": "auto",
            "padding-left": "10px",
            "padding-right": "10px",
        },
    )


def disclaimer_layout():
    return html.Div(
        [html.H5("This is not an official Government Maproom.")],
        id="disclaimer_panel",
        className="info",
        style={
            "position": "absolute",
            "width": "fit-content",
            "z-index": "1000",
            "height": "fit-content",
            "bottom": "0",
            "right": "0",
            "pointer-events": "auto",
            "padding-left": "10px",
            "padding-right": "10px",
        },
    )


def command_layout():
    return html.Div(
        [
            dcc.Input(id="geom_key", type="hidden"),
            dcc.Input(id="prob_thresh", type="hidden"),
            html.Div(
                [
                    help_layout(
                        "Mode",
                        "mode_label",
                        "The spatial resolution such as National, Regional, District or Pixel level",
                    ),
                    dcc.Dropdown(
                        id="mode",
                        clearable=False,
                    ),
                ],
                style={
                    "position": "relative",
                    "width": "120px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                },
            ),
            html.Div(
                [
                    help_layout(
                        "Issue",
                        "issue_label",
                        "The month in which the forecast is issued",
                    ),
                    dcc.Dropdown(
                        id="issue_month",
                        clearable=False,
                    ),
                ],
                style={
                    "position": "relative",
                    "width": "120px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                },
            ),
            html.Div(
                [
                    help_layout(
                        "Season", "season_label", "The rainy season being forecasted"
                    ),
                    dcc.Dropdown(
                        id="season",
                        clearable=False,
                    ),
                ],
                style={
                    "position": "relative",
                    "width": "95px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                },
            ),
            html.Div(
                [
                    help_layout(
                        "Year",
                        "year_label",
                        "The year whose forecast is displayed on the map",
                    ),
                    dcc.Dropdown(
                        id="year",
                        clearable=False,
                    ),
                ],
                style={
                    "position": "relative",
                    "width": "105px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                },
            ),
            html.Div(
                [
                    html.Label("Observations:"),
                    dcc.Dropdown(
                        id="obs_datasets",
                        clearable=False,
                        value="rain",
                    ),
                ],
                style={
                    "width": "95px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                },
            ),
            html.Div(
                [
                    help_layout(
                        "Severity",
                        "severity_label",
                        "The level of drought severity being targeted",
                    ),
                    dcc.Dropdown(
                        id="severity",
                        clearable=False,
                        options=[
                            dict(label="Low", value=0),
                            dict(label="Medium", value=1),
                            dict(label="High", value=2),
                        ],
                        value=0,
                    ),
                ],
                style={
                    "position": "relative",
                    "width": "95px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                },
            ),
            html.Div(
                [
                    help_layout(
                        "Frequency of triggered forecasts",
                        "frequency_label",
                        "The slider is used to set the frequency of forecast triggered",
                    ),
                    dcc.Slider(
                        id="freq",
                        min=5,
                        max=95,
                        step=5,
                        value=30,
                        marks={k: dict(label=f"{k}%") for k in range(10, 91, 10)},
                    ),
                ],
                style={
                    "position": "relative",
                    "width": "340px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                },
            ),
            html.Div(
                [
                    dcc.Loading(
                        html.A(
                            [
                                dbc.Button(
                                    "Gantt it!", color="info", id="gantt_button"
                                ),
                                dbc.Tooltip(
                                    "Gantt it!- Early action activities planning tool in a format of a Gantt chart",
                                    target="gantt_button",
                                    className="tooltiptext",
                                ),
                            ],
                            id="gantt",
                            target="_blank",
                        ),
                        type="dot",
                        parent_style={"height": "100%"},
                        style={"opacity": 0.2},
                    )
                ],
                style={
                    "position": "relative",
                    "width": "110px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                },
            ),
            html.Div(
                [
                    help_layout(
                        "Probability threshold",
                        "prob_label",
                        "To trigger at the selected frequency, trigger when the forecast probability of drought is at least this high.",
                    ),
                    html.Div(id='prob_thresh_text'),
                ],
                style={
                    "position": "relative",
                    "width": "1px", # force it to wrap
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                },
            ),
            dbc.Alert(
                "No forecast available for this month",
                color="danger",
                dismissable=True,
                is_open=False,
                id="forecast_warning",
                style={
                    "margin-bottom": "8px",
                },
            ),

        ],
        id="command_panel",
        className="info",
        style={
            "position": "absolute",
            "top": "10px",
            "right": "10px",
            "left": "230px",
            "z-index": "1000",
            "height": "fit-content",
            "pointer-events": "auto",
            "padding-left": "10px",
            "padding-right": "10px",
        },
    )


def table_layout():
    return html.Div(
        [
            html.Div(id="log"),
            dcc.Loading(
                [
                    html.Table(
                        id="table",
                        style={
                            "height": "650px",
                            "maxHeight": "650px",
                        },
                    ),
                ],
                type="dot",
                parent_style={"height": "100%"},
                style={"opacity": 0.2},
            ),
        ],
        className="info",
        style={
            "position": "absolute",
            "top": "110px",
            "right": "10px",
            "z-index": "1000",
            "height": "fit-content",
            "width": "600px",
            "pointer-events": "auto",
            "padding-left": "10px",
            "padding-right": "10px",
        },
    )


@dataclass
class ColorPair:
    fg: str # foreground
    bg: str # background


SEVERITY_COLORS = [
    ColorPair("black", "#fdfd96"),
    ColorPair("black", "#ffb347"),
    ColorPair("white", "#ff6961"),
]


