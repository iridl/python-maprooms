from dash import dcc
from dash import html
import dash_leaflet as dlf
import dash_leaflet.express as dlx
import dash_bootstrap_components as dbc
import uuid

SEVERITY_COLORS = ["#fdfd96", "#ffb347", "#ff6961"]


def app_layout():
    return html.Div(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Note")),
                    dbc.ModalBody(dcc.Markdown(id="modal-content"))
                ],
                id="modal",
                centered=True,
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle(id="validation-title")),
                    dbc.ModalBody(id="validation-body")
                ],
                id="validation-modal",
                centered=True,
            ),
            dcc.Location(id="location", refresh=False),
            control_layout(),
            html.Div([
                html.Div(map_layout(), id="lcol"),
                html.Div(table_layout(), id="rcol"),
            ], id="main_row"),
            html.Div(
                [html.H5("This is not an official Government Maproom.")],
                id="disclaimer_panel",
                className="info",
                style={
                    "position": "absolute",
                    "width": "fit-content",
                    "zIndex": "1000",
                    "height": "fit-content",
                    "bottom": "0",
                    "left": "0",
                    "pointerEvents": "auto",
                    "paddingLeft": "10px",
                    "paddingRight": "10px",
                },
            )
        ],
        id="toplevel",
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
                    dlf.GeoJSON(
                        options=dict(
                            color="rgb(49, 109, 150)",
                            fillColor="orange",
                            fillOpacity=0.1,
                            weight=2,
                        ),
                        id="outline",
                    ),
                    dlf.Marker(
                        [
                            dlf.Popup([
                                dcc.Loading(id="marker_popup", type="dot"),
                            ]),
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
        closePopupOnClick=False,
    )


def control(label, tool, ctrl, width=None, min_width='105px', grow=True):
    style = {
        "display": "inline-block",
        "padding": "10px",
        "verticalAlign": "middle",
    }

    if width is None:
        if grow:
            style["flexGrow"] = 1
        if min_width is not None:
            style["minWidth"] = min_width
    else:
        style['width'] = width

    return html.Div(
        [label_with_tooltip(label, tool), ctrl],
        style=style,
    )


def control_layout():
    return html.Div(
        className="info",
        style={'display': 'flex', 'flex-direction': 'column'},
        children=[
            html.Div(
                id="control_row",
                children=[
                    dcc.Store(id="geom_key"),
                    html.Div(
                        style={
                            'display': 'flex',
                            'flex-direction': 'column',
                            'padding-right': '5px',
                        },
                        children=[
                            html.H4("AA Design"),
                            html.Img(id="logo"),
                        ],
                    ),
                    control(
                        "Mode",
                        "The spatial resolution such as National, Regional, District or Pixel level",
                        dcc.Dropdown(
                            id="mode",
                            clearable=False,
                        ),
                    ),

                    control(
                        "Forecast",
                        "Which forecast to display on the map",
                        dcc.Dropdown(id="map_column", clearable=False, optionHeight=45),
                        # min_width='130px',
                    ),


                    control(
                        "Issue",
                        "The month in which the forecast is issued",
                        dcc.Dropdown(
                            id="issue_month",
                            clearable=False,
                        ),
                    ),

                    control(
                        "Season",
                        "The rainy season being forecasted",
                        dcc.Dropdown(
                            id="season",
                            clearable=False,
                        ),
                    ),

                    control(
                        "Year",
                        "The year whose forecast is displayed on the map",
                        dcc.Dropdown(
                            id="year",
                            clearable=False,
                        ),
                    ),

                    control(
                        "Severity",
                        "The level of drought severity being targeted",
                        dcc.Dropdown(
                            id="severity",
                            clearable=False,
                            options=[
                                dict(label="Low", value=0),
                                dict(label="Medium", value=1),
                                dict(label="High", value=2),
                            ],
                        ),
                    ),

                    control(
                        "Frequency of trigger events",
                        "The slider is used to set the frequency of the trigger",
                        dcc.Slider(
                            id="freq",
                            min=5,
                            max=95,
                            step=5,
                            marks={k: dict(label=f"{k}%") for k in range(10, 91, 10)},
                        ),
                        width="350px",
                    ),

                    control(
                        "Toggle",
                        "Toggle display of map and table",
                        dcc.Checklist(
                            ['Map', 'Table',],
                            ['Map', 'Table',],
                            id="fbf_display",
                            inputStyle={"margin-right": "5px"},
                            labelStyle={"display": "flex"}
                        ),
                        grow=False,
                    ),
                    control(
                        "CSV Validator",
                        "Validate a bad years or vulnerability CSV file",
                        dcc.Upload(dbc.Button("Validate"), id="upload"),
                        grow=False,
                    ),
                ],
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
    )


def table_layout():
    return [
        html.Div(
            [
                control(
                    "Reference dataset",
                    "Column that serves as the baseline. Other columns will be "
                    "scored by how well they match this one.",
                    dcc.Dropdown(
                        id="predictand",
                        clearable=False,
                        optionHeight=45,
                    ),
                ),
                control(
                    "Datasets",
                    "Other datasets to display in the table",
                    dcc.Dropdown(
                        id="predictors",
                        clearable=False,
                        multi=True,
                        optionHeight=45,
                    ),
                ),
                control(
                    "Include\nupcoming",
                    "If this is checked, data for the upcoming season "
                    "is included in the threshold calculation. "
                    "If unchecked, it is not included "
                    "in the calculation and its row in the table "
                    "is grayed out.",
                    dbc.Checkbox(
                        id="include_upcoming",
                    ),
                    width='100px',
                ),
            ],
            id="table_controls",
            style={"font-family": "Arial, Helvetica, sans-serif"},
        ),
        dcc.Loading(
            html.Div(id="table_container"),
            type="dot",
            parent_style={
                "flexGrow": 1,
                "minHeight": 0,
            },
        ),      
    ]
