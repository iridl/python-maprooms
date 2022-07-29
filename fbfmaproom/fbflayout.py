from dash import dcc
from dash import html
from dash import dash_table as table
import dash_leaflet as dlf
import dash_leaflet.express as dlx
import dash_bootstrap_components as dbc
import uuid

SEVERITY_COLORS = ["#fdfd96", "#ffb347", "#ff6961"]


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


def label_with_tooltip(label, tooltip):
    id_name = make_id()
    return html.Div(
        [
            html.Label(f"{label}:", id=id_name, style={"cursor": "pointer"}),
            dbc.Tooltip(
                f"{tooltip}",
                target=id_name,
                className="tooltiptext",
            ),
        ]
    )


def make_id():
    return str(uuid.uuid4())


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
                        dlf.TileLayer(opacity=0.8, id="raster_layer"),
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
                id="raster_colorbar",
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
        # Override dash-leaflet's silly default that causes it to
        # waste time loading the basemap for western Europe when the
        # page first loads.
        center=None,
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
            "zIndex": "1000",
            "height": "fit-content",
            "pointerEvents": "auto",
            "paddingLeft": "10px",
            "paddingRight": "10px",
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
            "zIndex": "1000",
            "height": "fit-content",
            "bottom": "0",
            "right": "0",
            "pointerEvents": "auto",
            "paddingLeft": "10px",
            "paddingRight": "10px",
        },
    )


def command_layout():
    return html.Div(
        [
            dcc.Store(id="geom_key"),
            dcc.Input(id="map_column", type="hidden", value="pnep"),
            html.Div(
                [
                    label_with_tooltip(
                        "Mode",
                        "The spatial resolution such as National, Regional, District or Pixel level",
                    ),
                    dcc.Dropdown(
                        id="mode",
                        clearable=False,
                    ),
                ],
                style={
                    "position": "relative",
                    "width": "105px",
                    "display": "inline-block",
                    "padding": "10px",
                    "verticalAlign": "top",
                },
            ),
            html.Div(
                [
                    label_with_tooltip(
                        "Issue",
                        "The month in which the forecast is issued",
                    ),
                    dcc.Dropdown(
                        id="issue_month",
                        clearable=False,
                    ),
                ],
                style={
                    "position": "relative",
                    "width": "85px",
                    "display": "inline-block",
                    "padding": "10px",
                    "verticalAlign": "top",
                },
            ),
            html.Div(
                [
                    label_with_tooltip(
                        "Season", "The rainy season being forecasted"
                    ),
                    dcc.Dropdown(
                        id="season",
                        clearable=False,
                    ),
                ],
                style={
                    "position": "relative",
                    "width": "85px",
                    "display": "inline-block",
                    "padding": "10px",
                    "verticalAlign": "top",
                },
            ),
            html.Div(
                [
                    label_with_tooltip(
                        "Year",
                        "The year whose forecast is displayed on the map",
                    ),
                    dcc.Dropdown(
                        id="year",
                        clearable=False,
                    ),
                ],
                style={
                    "position": "relative",
                    "width": "85px",
                    "display": "inline-block",
                    "padding": "10px",
                    "verticalAlign": "top",
                },
            ),
            html.Div(
                [
                    label_with_tooltip(
                        "Severity",
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
                    "verticalAlign": "top",
                },
            ),
            html.Div(
                [
                    label_with_tooltip(
                        "Frequency of trigger events",
                        "The slider is used to set the frequency of the trigger",
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
                    "verticalAlign": "top",
                },
            ),
            dbc.Alert(
                "No data available for selected month and year",
                color="danger",
                dismissable=True,
                is_open=False,
                id="forecast_warning",
                style={
                    "marginBottom": "8px",
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
            "zIndex": "1000",
            "height": "fit-content",
            "pointerEvents": "auto",
            "paddingLeft": "10px",
            "paddingRight": "10px",
        },
    )


def table_layout():
    return html.Div(
        [
            html.Div(id="log"),
            html.Div(
                [
                    label_with_tooltip(
                        "Baseline observations",
                        "Column that serves as the baseline. Other columns will be "
                        "scored by how well they predict this one.",
                    ),
                    dcc.Dropdown(
                        id="predictand",
                        clearable=False,
                    ),
                ],
                style={
                    "display": "inline-block",
                    "padding": "10px",
                    "verticalAlign": "top",
                    "width": "30%",
                },
            ),
            html.Div(
                [
                    label_with_tooltip(
                        "Predictors",
                        "Other datasets to display in the table"
                    ),
                    dcc.Dropdown(
                        id="predictors",
                        clearable=False,
                        multi=True,
                    ),
                ],
                style={
                    "display": "inline-block",
                    "padding": "10px",
                    "verticalAlign": "top",
                    "width": "70%",
                },
            ),

            dcc.Loading(
                [
                    html.Div(id="table_container", style={"height": "100%"})
                ],
                type="dot",
                parent_style={
                    "position": "absolute",
                    "top": "80px",
                    "bottom": "10px",
                    "left": "10px",
                    "right": "10px",
                },
            ),
        ],
        className="info",
        style={
            "position": "absolute",
            "top": "110px",
            "right": "10px",
            "zIndex": "1000",
            "bottom": "50px",
            "width": "600px",
            "pointerEvents": "auto",
            "paddingLeft": "10px",
            "paddingRight": "10px",
        },
    )
