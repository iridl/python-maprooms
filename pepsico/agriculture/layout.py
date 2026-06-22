from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_leaflet as dlf
import dash_leaflet.express as dlx
from fieldsets import Block, Select, PickPoint, Month, Number
from dash_extensions.javascript import arrow_function, assign
import json 
from globals_ import GLOBAL_CONFIG

# -----------------------------
# Model, year, variety and scenario options
# -----------------------------
MODELS = [
    {"value": "CanESM2.CanRCM4", "label": "CanESM2 + CanRCM4"},
    {"value": "CanESM2.CRCM5-UQAM", "label": "CanESM2 + CRCM5-UQAM"},
    {"value": "MPI-ESM-LR.CRCM5-UQAM", "label": "MPI-ESM-LR + CRCM5-UQAM"},
]

HISTORICAL_YEARS = [
    {"label": "2006-2010", "value": "2006-2010"},
]
PROJECTED_YEARS = [
    {"label": "2021-2025", "value": "2021-2025"},
    {"label": "2026-2030", "value": "2026-2030"},
    {"label": "2031-2035", "value": "2031-2035"},
    {"label": "2036-2040", "value": "2036-2040"},
    {"label": "2041-2045", "value": "2041-2045"},
    {"label": "2046-2050", "value": "2046-2050"}      
]

VARIETY = [
    {"label": "Atlantic", "value": "Atlantic"},
    {"label": "RH-02", "value": "1867"},
    {"label": "RH-03", "value": "2053"},
    {"label": "RH-05", "value": "2137"},
    {"label": "RH-06", "value": "2215"},
    {"label": "RH-13", "value": "2312"},
] 
PLANTING = [
    {"label": "No adaptation", "value": "PDhist"},
    {"label": "With adaptation", "value": "PDy0n30"},
]
SCENARIOS= [
    {"label": "RCP 4.5", "value": "rcp45"},
    {"label": "RCP 8.5", "value": "rcp85"},     
]

# -----------------------------
# Interface color palette
# -----------------------------
IRI_BLUE = "rgb(25,57,138)"
IRI_GRAY = "rgb(113,112,116)"
LIGHT_GRAY = "#eeeeee"

# -----------------------------
# Color classes and styles for map initialization
# -----------------------------
classes = [0, 10, 20, 50, 100, 200, 500, 1000]
colorscale = ["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C",
              "#FC4E2A", "#E31A1C", "#BD0026", "#800026"]

style_default = dict(weight=0.8, opacity=1, color="white", dashArray="3", fillOpacity=0.7)

# -----------------------------
# Load base GeoJSON
# -----------------------------
with open("data/shapes/pepsico.json") as f:
        data = json.load(f)

# Add HARWT and name to feature properties
for feature in data["features"]:
        feature["properties"]["NAME"] = feature["properties"].get(
            "CDNAME", feature["properties"].get("NAME", "Unknown")
        )
        feature["properties"]["STATE_NAME"] = feature["properties"].get(
            "PRNAME", feature["properties"].get("STATE_NAME", "Unknown")
        )

# -----------------------------
# Store for dynamic data
# -----------------------------
colorbar = html.Div(id="colorbar")  # empty on init, populated by callback

# -----------------------------
# JS style handler for GeoJSON - links county shapes to choropleth coloring
# -----------------------------
style_handle =  assign(
        """
        function(feature, context) {
            const { classes, colorscale, style, colorProp } = context.hideout;
            const value = feature.properties[colorProp];
            let fillColor = colorscale[0];

            for (let i = 0; i < classes.length - 1; i++) {
                if (value >= classes[i] && value < classes[i + 1]) {
                    fillColor = colorscale[i];
                    break;
                }
            }

            if (value >= classes[classes.length - 1]) {
                fillColor = colorscale[colorscale.length - 1];
            }

            return { ...style, fillColor };
        }
        """
)

# -----------------------------
# GeoJSON layer (initially empty, filled by callback)
# -----------------------------
geojson = dlf.GeoJSON(
    data={"type": "FeatureCollection", "features": []},
    style=style_handle,
    hoverStyle=arrow_function(dict(weight=2, color="#666", dashArray="")),
    zoomToBounds=False,
    hideout=dict(colorscale=colorscale, classes=classes, style=style_default, colorProp="HARWT"),
    id="geojson"
)

# -------------------------------------------------
# Main application layout
# -------------------------------------------------
def app_layout():
    return dbc.Container(
        [
            # Component for URL routing in Dash
            dcc.Location(id="location", refresh=True),
            
            # Top navbar with main controls
            navbar_layout(),

            # Main container with two columns: controls and map/results
            dbc.Row(
                [
                    # Controls column
                    dbc.Col(
                        controls_layout(),
                        sm=12, md=4,
                        style={
                            "background-color": "white", "border-style": "solid",
                            "border-color": LIGHT_GRAY,
                            "border-width": "0px 1px 0px 0px",
                        },
                    ),
                    # Map and results column
                    dbc.Col(
                        [
                            # Map row
                            dbc.Row(
                                [
                                     dbc.Col(
                                         map_layout(),
                                         width=12,
                                     ),
                                ],
                            ),
                            # Results and charts row
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


# -------------------------------------------------
# Tooltip helper layout for buttons
# -------------------------------------------------
def help_layout(buttonname, id_name, message):
    return html.Div(
        [
            html.Label(
                f"{buttonname}:", id=id_name,
                style={"cursor": "pointer","font-size": "100%","padding-left":"3px"},
            ),
            # Tooltip with help message
            dbc.Tooltip(f"{message}", target=id_name, className="tooltiptext"),
        ]
    )


# -------------------------------------------------
# Top navbar: region, scenario, model, variable, period selectors, etc
# -------------------------------------------------
def navbar_layout():
    return dbc.Nav(
        [
            # App brand / logo
            html.A(
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.NavbarBrand(" Agriculture Index App", 
                                            className="ml-2",
                                            style={
                                                    "font-size": "2rem",
                                                    "font-weight": "bold",
                                                    "text-align": "center",
                                                    "color": "white",
                                                }
                                            ),
                            width=12,
                            
                        ),
                    ],
                    align="center",
                    id='navbar-brand' ,
                    style={
                        "padding-top": "1rem",
                        "padding-bottom": "1rem",
                    },
                ),
            ),
            # Control block: scenario, model, variable, month, year selectors
            Block("",
                html.Div(
                    [ 
                    Block("Variable", Select(
                        id="variable",
                        options=[
                            "HARWT",
                        ],
                        labels=[
                            "Harvested Weight",
                        ],
                        init=0,
                    )),
                    # Variety selection
                    Block("Variety", Select(id="variety", 
                        options=[m["value"] for m in VARIETY],
                            labels=[m["label"] for m in VARIETY],
                        init=0
                    )),
                
                # Data view type
                Block("Data View", 
                    Select(id="data_type", 
                        options=[
                            "historical",
                            "projected",
                            "mean_change",
                            "percentage_change",
                            "direction_change",
                            "yield_change_index",
                            "stress_simple"

                        ],
                        labels=[
                            "Historical",
                            "Projected",
                            "Mean change",
                            "Percentage change (Δ%)",
                            "Direction of change",
                            "Yield Change Index",
                            "Stress Simple Index"

                        ],
                    ),
                ),
                ],
                    style={"display": "inline-flex", "gap": "5px","margin-left": "5px", "vertical-align": "top", "align-items": "flex-start"}   
                ),
                html.Div(
                    [ 
                        Block("Model", Select(id="model", 
                            options=[m["value"] for m in MODELS],
                            labels=[m["label"] for m in MODELS]
                        )),
                        Block("Planting", Select(
                            id="planting",
                            options=[m["value"] for m in PLANTING],
                            labels=[m["label"] for m in PLANTING],
                            init=0,
                        )),                        
                        Block("Scenario", Select(
                            id="scenario",
                            options=[m["value"] for m in SCENARIOS],
                            labels=[m["label"] for m in SCENARIOS],
                            init=0,
                        )),
                    ],
                    id="model_container",
                    style={"display": "inline-flex", "gap": "5px", "margin-left": "10px", "vertical-align": "top", "align-items": "flex-start"}   
                ),
                # Projected year selection
                html.Div( 
                    Block("Period", 
                        Select(id="period_years",
                                options=[
                                    "2006-2010",
                                    "2046-2050",
                                ],
                                labels=[
                                    "Historical (2006-2010)",
                                    "Projected (2046-2050)"
                                ],
                                )
                    ),
                    id="period_container",
                    style={"display": "inline-block", "margin-left": "20px", "vertical-align": "top"}   
                ),
                # Anomaly / comparison period selectors (Dataset 1 vs Dataset 2)
                html.Div( 
                    Block("", 
                        "  ",
                        Block("Dataset 1", 
                            html.Div(
                                children=[
                                    Select(id="dataset1",
                                        options=[m["value"] for m in MODELS],
                                        labels=[m["label"] for m in MODELS]
                                        ),
                                    " ",
                                    Select(
                                        id="dataset1_planting",
                                        options=[m["value"] for m in PLANTING],
                                        labels=[m["label"] for m in PLANTING],
                                        init=0,
                                        ),
                                    " ",
                                    Select(
                                        id="dataset1_scenario",
                                        options=[m["value"] for m in SCENARIOS],
                                        labels=[m["label"] for m in SCENARIOS],
                                        init=0,
                                        ),
                                    " ",
                                    Select(id="dataset1_years",
                                        options=[m["value"] for m in PROJECTED_YEARS],
                                        labels=[m["label"] for m in PROJECTED_YEARS]
                                            ),
                                ],
                                id="dataset1_container",
                            ),
                        ),
                        " ",
                        Block("Dataset 2", 
                            
                            " ",
                            html.Div(
                                children=[
                                    Select(id="dataset2",
                                        options=['historical']+[m["value"] for m in MODELS],
                                        labels=['Historical']+[m["label"] for m in MODELS]
                                    ),
                                    html.Div(
                                        children=[
                                        Select(
                                            id="dataset2_planting",
                                            options=[m["value"] for m in PLANTING],
                                            labels=[m["label"] for m in PLANTING],
                                            init=0,
                                            ),
                                        " ",
                                        Select(
                                            id="dataset2_scenario",
                                            options=[m["value"] for m in SCENARIOS],
                                            labels=[m["label"] for m in SCENARIOS],
                                            init=0,
                                            ),
                                        ],
                                        id="dataset2_plant-scen_container"
                                    ),
                                    " ",
                                    Select(id="dataset2_years",
                                    options=[m["value"] for m in HISTORICAL_YEARS],
                                    labels=[m["label"] for m in HISTORICAL_YEARS]
                                        ),
                                ],
                                id="dataset2_container",
                            ),

                        ),
                    ),
                    id="anomaly_period_container",
                    style={"display": "inline-block", "margin-left": "20px", "vertical-align": "top"}   
                ),
                # Store used to pass anomaly period values between components
                dcc.Store(id="anomaly_period_values"),

            ),
            # Alert for map errors
            dbc.Alert(
                "Something went wrong",
                color="danger",
                dismissable=True,
                is_open=False,
                id="map_warning",
                style={"margin-bottom": "8px"},
            ),
        ],
        style={"display": "flex", "flex-direction": "column","background-color": IRI_GRAY},
    )


# -------------------------------------------------
# Country border overlay (USA + Canada)
# -------------------------------------------------
with open("data/shapes/countries_50m.json") as f:
    countries = json.load(f)

features = [f for f in countries["features"] if f["properties"]["adm0_a3"] in ["USA", "CAN"]]
geojson_borders = {
    "type": "FeatureCollection",
    "features": features
}

borders_layer = dlf.GeoJSON(
    data=geojson_borders,
    id="country-borders",
    zoomToBounds=False,
    options=dict(style=dict(color="black", weight=1.2, fill=False, fillOpacity=0)),
    interactive=True,
)


# -----------------------------
# Basemap tile layers
# -----------------------------
terrenos = [
    dlf.TileLayer(id="osm", url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                 attribution="© OpenStreetMap contributors"),
    dlf.TileLayer(id="carto_light", url="https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
                 attribution="© CARTO"),
    dlf.TileLayer(id="carto_dark", url="https://cartodb-basemaps-a.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png",
                 attribution="© CARTO"),
    dlf.TileLayer(id="carto_voyager", url="https://cartodb-basemaps-a.global.ssl.fastly.net/rastertiles/voyager/{z}/{x}/{y}.png",
                 attribution="© CARTO"),
    dlf.TileLayer(id="stamen", url="https://stamen-tiles.a.ssl.fastly.net/toner/{z}/{x}/{y}.png",
                 attribution="Map tiles by Stamen Design"),
    dlf.TileLayer(id="opentopomap", url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                 attribution="© OpenTopoMap contributors"),
    dlf.TileLayer(id="esri_world_imagery",
                 url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                 attribution="Tiles © Esri"),
    dlf.TileLayer(id="esri_world_topo",
                 url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
                 attribution="Tiles © Esri"),
    dlf.TileLayer(id="esri_world_street",
                 url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                 attribution="Tiles © Esri")
]

# -----------------------------
# Layer switcher control
# -----------------------------
layers_control = dlf.LayersControl(
    [*[
        dlf.BaseLayer(layer, name=layer.id, checked=(i == 0))
        for i, layer in enumerate(terrenos)
    ],
    dlf.Overlay(borders_layer, name="USA/CAN Borders", id="overlay-borders", checked=True)],
    position="topleft",
    id="layers-control"
)

# -----------------------------
# Hover / click info box
# -----------------------------
def get_info(feature=None):
    if not feature:
        return [html.P("Hover or click over a municipality")]
    return [
        html.B(f"Code: {feature['properties']['id']}"), html.Br(),
        html.B(f"{feature['properties']['NAME']} , {feature['properties']['STATE_NAME']}"), html.Br(),
        f"HARWT: {feature['properties']['HARWT']}"
    ]

info = html.Div(
    children=get_info(),
    id="info",
    style={
        "position": "absolute",
        "top": "200px",
        "right": "15px",
        "zIndex": 1000,
        "background": "white",
        "padding": "6px",
        "borderRadius": "4px",
        "boxShadow": "0px 0px 3px rgba(0,0,0,0.3)",
        
        # Responsive sizing
        "fontSize": "0.8rem",
        "width": "fit-content",
        "maxWidth": "30vw",
        "overflow": "auto",
    }
)


# -------------------------------------------------
# Sidebar controls layout: description and instructions
# -------------------------------------------------
def controls_layout():
    return dbc.Container(
        [
            html.H5(["Harvested Weight index monitoring"]),
            html.P(
                """
                This Maproom displays projected change of key harvested weight index.
                """
            ),
            # Map description loaded dynamically via callback
            dcc.Loading(html.P(id="map_description"), type="dot"),
            html.P(
                """
                Use the controls in the top banner to choose other variables, models,
                projected years and reference to compare with.
                """
            ),
            html.P(
                """
                Click the map (or enter coordinates) to show historical time
                series for this variable of this model, followed by a plume of
                possible projected scenarios.
                """
            ),
            dcc.Loading(html.P(id="map_description_var"), type="dot"),

        ],
        fluid=True, className="scrollable-panel",
        style={"padding-bottom": "1rem", "padding-top": "1rem"},
    )


# -------------------------------------------------
# Main map layout: title, colorbar, marker and tile layers
# -------------------------------------------------
def map_layout():

    return dbc.Container(
        [
            # Map title (populated by callback)
            html.H5(
                id="map_title",
                style={
                    "text-align":"center", "border-width":"1px",
                    "margin-top":"3px", "margin-bottom":"3px",
                },
            ),
            # Map component with GeoJSON, layer control and colorbar
            dcc.Loading(
                        dlf.Map(
                            children=[
                                    dlf.TileLayer(), 
                                    geojson, 
                                    layers_control, 
                                    colorbar,                                     
                                    ],
                            center=[45, -95],
                            zoom=4,
                            minZoom=3,
                            maxZoom=8,
                            maxBounds=[[5, -170], [85, -10]],
                            maxBoundsViscosity=1.0,
                            style={"height": "90vh", "width": "100%"},
            ),
             type="dot", 
             overlay_style={"visibility":"visible", "opacity": .5, "backgroundColor": "white", 
                            }, 
             ),
             info,
            # Hidden div used as dummy callback output
            html.Div(id="dummy-output", style={"display": "none"}), 
        ],
        fluid=True,
    )


# -------------------------------------------------
# Results layout: time series chart and data download
# -------------------------------------------------
def results_layout():
    return dbc.Tabs([
        dbc.Tab([
            # Toolbar row: graph type selector (left) and download button (right)
            html.Div(
                dbc.Row(
                    [
                        # Left side: graph controls
                        dbc.Col(
                            html.Div(
                                [
                                    html.Span("Graph type", style={"margin-left": "5px","margin-right": "5px"}),
                                    Select(
                                        id="graph_type",
                                        options=["bars_group","bars_stack", "lines","lines+markers","markers"],
                                        labels= ["bars group","bars stack","lines","lines+dots","dots"],
                                        init=0,
                                    ),
                                    html.Span("Planting", style={"margin-left": "10px","margin-right": "5px"}),
                                    Select(
                                        id="graph_planting",
                                        options=[m["value"] for m in PLANTING],
                                        labels=[m["label"] for m in PLANTING],
                                        ),
                                    html.Div(
                                        [
                                            html.Span("Scenario", style={"margin-left": "10px","margin-right": "5px"}, id="graph_sce_text"),
                                            Select(
                                                id="graph_scenario",
                                                options=[m["value"] for m in SCENARIOS],
                                                labels=[m["label"] for m in SCENARIOS],
                                                init=0,
                                            ),
                                        ],
                                        id="graph_scenario_container",
                                        style={"display": "flex", "align-items": "center"}
                                    ),

                                ],
                                style={"display": "flex", "align-items": "center"}
                            ),
                            width="auto"
                        ),

                        # Right side: download button and component
                        dbc.Col(
                            [
                                dbc.Button("Download data in graph", id="btn_csv", size="sm"),
                                dcc.Download(id="download-dataframe-csv"),
                            ],
                            width="auto",
                            className="text-end"
                        ),
                    ],
                    className="align-items-center justify-content-between"
                ),
                style={"width": "100%"}
            ),

            # Chart area
            html.Div([dbc.Spinner(dcc.Graph(id="local_graph"))]),
        ], label="Local History and Projections")
    ])
