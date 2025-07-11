import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import dash_leaflet as dlf


IRI_BLUE = "rgb(25,57,138)"
IRI_GRAY = "rgb(113,112,116)"
LIGHT_GRAY = "#eeeeee"


def app_layout_1(navbar, description, map, local):
    """
    Layout with controls in navbar at the top,
    2 columns: a narrow left one with descriptions and right one with
    2 rows: map on top of local graph
    """
    return dbc.Container(
        [
            dcc.Location(id="location", refresh=True),
            navbar,
            dbc.Row(
                [
                    dbc.Col(
                        description, sm=12, md=4,
                        style={
                            "background-color": "white", "border-style": "solid",
                            "border-color": LIGHT_GRAY,
                            "border-width": "0px 1px 0px 0px",
                        },
                    ),
                    dbc.Col(
                        [
                            dbc.Row([dbc.Col(
                                map, width=12, style={"background-color": "white"},
                            )]),
                            dbc.Row([
                                dbc.Col(
                                    local, width=12,
                                    style={
                                        "background-color": "white",
                                        "min-height": "100px",
                                        "border-style": "solid",
                                        "border-color": LIGHT_GRAY,
                                        "border-width": "1px 0px 0px 0px",
                                    },
                                )],
                            ),
                        ],
                        sm=12, md=8, style={"background-color": "white"},
                    ),
                ],
            ),
        ],
        fluid=True, style={"padding-left": "0px", "padding-right": "0px"},
    )


def navbar_layout(title, *elems):
    """
    A title followed by a line-up of controls
    and a dbc.Alert
    """
    return dbc.Nav(
        [
            html.A(dbc.Row(
                [dbc.Col(dbc.NavbarBrand(title, className="ml-2"))],
                align="center", style={"padding-left": "5px", "color": "white"},
            ))
        ] + [
            elems[i] for i in range(len(elems))
        ] + [
            dbc.Alert(
                "Something went wrong",
                color="danger",
                dismissable=True,
                is_open=False,
                id="map_warning",
                style={"margin-bottom": "8px"},
            ),
        ],
        style={"background-color": IRI_GRAY},
    )


def description_layout(title, subtitle, *elems):
    """
    An H5 title, a subtitle sentence followed by a series of paragraphs
    or other html elements
    """
    return dbc.Container(
        [
            html.H5([title]), html.P(subtitle),
            dcc.Loading(html.P(id="map_description"), type="dot"),
        ] + [
            elems[i] for i in range(len(elems))
        ],
        fluid=True, className="scrollable-panel",
        style={"padding-bottom": "1rem", "padding-top": "1rem"},
    )


def map_layout(
    default_zoom,
    layers_control_position="topleft", scale_control_position="bottomright",
    cb_nTicks=9, cb_opacity=1, cb_tooltip=True,
    cb_position="topright", cb_width=10, cb_height=300,
):
    """
    A dlf map topped with and H5 title,
    positioned layers control, scale and colorbar,
    a dlf.Marker for local selection,
    and default colorbar options
    """
    return dbc.Container(
        [
            dcc.Loading(html.H5(
                id="map_title",
                style={
                    "text-align":"center", "border-width":"1px",
                    "border-style":"solid", "border-color":"grey",
                    "margin-top":"3px", "margin-bottom":"3px",
                },
            ), type="dot"),
            dcc.Loading(dlf.Map(
                [
                    dlf.LayersControl(
                        id="layers_control", position=layers_control_position
                    ),
                    dlf.LayerGroup(
                        [dlf.Marker(id="loc_marker", position=(0, 0))],
                        id="layers_group"
                    ),
                    dlf.ScaleControl(
                        imperial=False, position=scale_control_position
                    ),
                    dlf.Colorbar(
                        id="colorbar",
                        nTicks=cb_nTicks,
                        opacity=cb_opacity,
                        tooltip=cb_tooltip,
                        position=cb_position,
                        width=cb_width,
                        height=cb_height,
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
                zoom=default_zoom,
                style={"width": "100%", "height": "50vh"},
            ), type="dot"),
        ],
        fluid=True,
    )


def local_single_tabbed_layout(label, download_button=False):
    """
    Single tabbed local graph with or without a download data button
    """
    if download_button:
        button = [html.Div([
            dbc.Button("Download local data", id="btn_csv", size="sm"),
            dcc.Download(id="download-dataframe-csv"),
        ], className="d-grid justify-content-md-end")]
    else:
        button = []
    return dbc.Tabs([dbc.Tab(
        button + [
            html.Div([dbc.Spinner(dcc.Graph(id="local_graph"))]),
        ],
        label=label)])


def help_layout(buttonname, id_name, message):
    """
    Can be tagged to a control to have it display a helpful tooltip
    when moused over
    """
    return html.Div(
        [
            html.Label(
                f"{buttonname}:", id=id_name,
                style={"cursor": "pointer","font-size": "100%","padding-left":"3px"},
            ),
            dbc.Tooltip(f"{message}", target=id_name, className="tooltiptext"),
        ]
    )