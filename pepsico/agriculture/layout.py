from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_leaflet as dlf
import dash_leaflet.express as dlx
from fieldsets import Block, Select, PickPoint, Month, Number

from globals_ import GLOBAL_CONFIG
import json 
import math

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

# Colores de la interfaz
IRI_BLUE = "rgb(25,57,138)"
IRI_GRAY = "rgb(113,112,116)"
LIGHT_GRAY = "#eeeeee"

# -----------------------------
# Funciones de clases y colores
# -----------------------------
def calcular_base(min_val, max_val, n=8):
    rango = max_val - min_val
    step_ideal = rango / (n - 1)
    exp = math.floor(math.log10(step_ideal))
    base = 10 ** exp
    multiplo = round(step_ideal / base)
    base = multiplo * base
    return max(base, 1)

def generar_clases(min_val, max_val, n=8, base=1000):
    min_val = int(min_val)
    max_val = int(max_val)
    lo = (min_val // base) * base
    hi = (max_val // base) * base
    if hi <= lo:
        return [lo + i * base for i in range(n)]
    raw_step = (hi - lo) / (n - 1)
    clases = [int(round(lo + raw_step * i) // base * base) for i in range(n)]
    for i in range(1, n):
        if clases[i] <= clases[i-1]:
            clases[i] = clases[i-1] + base
    clases[-1] = hi
    for i in range(n-2, -1, -1):
        if clases[i] >= clases[i+1]:
            clases[i] = clases[i+1] - base
    if clases[0] < lo:
        return [lo + i * base for i in range(n)]
    return clases

# -----------------------------
# Colores y estilo
# -----------------------------
classes = [0, 10, 20, 50, 100, 200, 500, 1000]
colorscale = ["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C",
              "#FC4E2A", "#E31A1C", "#BD0026", "#800026"]

style_default = dict(weight=0.8, opacity=1, color="white", dashArray="3", fillOpacity=0.7)

# -----------------------------
# Colorbar
# -----------------------------
ctg = ["{}".format(cls) for cls in classes[:-1]] + ["{}+".format(classes[-1])]


# -----------------------------
# Cargar geojson base
# -----------------------------
with open("data/shapes/pepsico.json") as f:
        data = json.load(f)

    # Agregar HARWT y nombre a properties
for feature in data["features"]:
        fid = str(feature["properties"].get("id", ""))
        feature["properties"]["NAME"] = feature["properties"].get(
            "CDNAME", feature["properties"].get("NAME", "Unknown")
        )
        feature["properties"]["STATE_NAME"] = feature["properties"].get(
            "PRNAME", feature["properties"].get("STATE_NAME", "Unknown")
        )

# -----------------------------
# Store para manejar datos dinámicos
# -----------------------------
store = dcc.Store(id="geojson-store")
colorbar = html.Div(id="colorbar")  # vacío al inicio, se llenará por callback

from dash_extensions.javascript import arrow_function, assign
# -----------------------------
# Estilo JS para geojson
# -----------------------------
# style_handle = assign("""
# function(feature, context){
#     const {classes, colorscale, style, colorProp} = context.hideout;
#     const value = feature.properties[colorProp];
#     let fillColor = colorscale[0];
#     for (let i = 0; i < classes.length; ++i) {
#         if (value > classes[i]) fillColor = colorscale[i];
#     }
#     return {...style, fillColor: fillColor};
# }
# """)

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

draw_control = html.Script("""
    function initDrawControl(map){
        var drawnItems = new L.FeatureGroup();
        map.addLayer(drawnItems);

        var drawControl = new L.Control.Draw({
            edit: { featureGroup: drawnItems }
        });
        map.addControl(drawControl);

        map.on(L.Draw.Event.CREATED, function (e) {
            var layer = e.layer;
            drawnItems.addLayer(layer);

            // Guardar geometría para Dash
            const geo = layer.toGeoJSON();
            window.dash_clientside.callback_context.triggered[0].geojson = geo;
        });
    }
""")



# -----------------------------
# Colores y estilo
# -----------------------------
#classes = [0, 10, 20, 50, 100, 200, 500, 1000]
#colorscale = ["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C",
#              "#FC4E2A", "#E31A1C", "#BD0026", "#800026"]

style_default = dict(weight=0.8, opacity=1, color="white", dashArray="3", fillOpacity=0.7)

# GeoJSON inicial vacío
geojson = dlf.GeoJSON(
    data={"type": "FeatureCollection", "features": []},
    style=style_handle,
    hoverStyle=arrow_function(dict(weight=2, color="#666", dashArray="")),
    zoomToBounds=False,
    hideout=dict(colorscale=colorscale, classes=classes, style=style_default, colorProp="HARWT"),
    id="geojson"
)

# -------------------------------------------------
# Layout principal de la aplicación
# -------------------------------------------------
def app_layout():
    return dbc.Container(
        [
            # Componente para manejo de rutas en Dash
            dcc.Location(id="location", refresh=True),
            
            # Navbar con controles principales
            navbar_layout(),

            # Contenedor principal con dos columnas: controles y mapa/resultados
            dbc.Row(
                [
                    # Columna de controles
                    dbc.Col(
                        controls_layout(),
                        sm=12, md=4,
                        style={
                            "background-color": "white", "border-style": "solid",
                            "border-color": LIGHT_GRAY,
                            "border-width": "0px 1px 0px 0px",
                        },
                    ),
                    # Columna de mapa y resultados
                    dbc.Col(
                        [
                            # Fila del mapa
                            dbc.Row(
                                [
                                     dbc.Col(
                                         map_layout(),
                                         width=12,
                                         #style={"background-color": "white"},
                                     ),
                                ],
                            ),
                            # Fila de resultados y gráficos
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
# Layout de ayuda para mostrar tooltips en los botones
# -------------------------------------------------
def help_layout(buttonname, id_name, message):
    return html.Div(
        [
            html.Label(
                f"{buttonname}:", id=id_name,
                style={"cursor": "pointer","font-size": "100%","padding-left":"3px"},
            ),
            # Tooltip con mensaje de ayuda
            dbc.Tooltip(f"{message}", target=id_name, className="tooltiptext"),
        ]
    )


# -------------------------------------------------
# Navbar superior con selección de región, escenario, modelo, variable y periodos
# -------------------------------------------------
def navbar_layout():
    return dbc.Nav(
        [
            # Logo o marca de la aplicación
            html.A(
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.NavbarBrand("Agriculture Index App", 
                                            className="ml-2",
                                            style={
                                                    "font-size": "2rem",      # letra más grande
                                                    "font-weight": "bold",    # opcional: más énfasis
                                                    "text-align": "center",   # centrado horizontal
                                                    "color": "white",
                                                }
                                            ),
                            width=12,
                            
                        ),
                    ],
                    align="center",
                    id='navbar-brand' ,
                    style={
                        "padding-top": "1rem",       # margen superior
                        "padding-bottom": "1rem",    # margen inferior
                    },
                ),
            ),
            # Salto de línea explícito
            #html.Br(),
            # Selección de región
            # dbc.Row(
            #     dbc.Col(
            #         Block("Region",
            #             Select(
            #                 id="region",
            #                 #options=["SAMER", "SASIA", "Thailand", "US-CA"],
            #                 # labels=[
            #                 #     "South America",
            #                 #     "South Asia",
            #                 #     "Thailand",
            #                 #     "United States and Canada",
            #                 # ],
            #                 options=["USA", "CAN"],
            #                 labels=["United States","Canada"],
            #                 #init=1,
            #             ),
            #             width=12,
            #         ),
            #         style={"margin-left": "0.5rem","margin-bottom": "1rem"},
            #     ),
            # ),
            # Herramienta de selección de coordenadas
            #PickPoint(width="8em"),
            #Bloque de envío con escenarios, modelos, variables, meses y años
            Block("",
                # Block("Region", Select(
                #     id="region",
                #     options=["USA", "CAN"],
                #     labels=["United States","Canada"],
                #     init=0,
                # )),
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
                    # Selección de estación (mes inicial y final)
                    # Block("Season",
                    #     Month(id="start_month", default="Jan"),
                    #     "-",
                    #     Month(id="end_month", default="Mar"),
                    # ),
                    # Selección de referncias
                    Block("Variety", Select(id="variety", 
                        options=[m["value"] for m in VARIETY],
                            labels=[m["label"] for m in VARIETY],
                        init=0
                    )),
                
                # Data type
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
                # Selección de años proyectados
                html.Div( 
                    Block("Period", 
                        Select(id="period_years",
                                options=[
                                    "2006-2010",
                                    "2046-2050",
                                ],
                                labels=[
                                    "Historical (2006-2010)",
                                    "Prohected (2046-2050)"
                                ],
                                )
                    ),
                    id="period_container",
                    style={"display": "inline-block", "margin-left": "20px", "vertical-align": "top"}   
                ),
                html.Div( 
                    Block("", 
                        "  ",
                        Block("Dataset 1", 
                        #PROJECTED_YEARS
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
                                #style={"display": "flex", "flex-direction": "column"} 
                            ),
                        ),
                        " ",
                        Block("Dataset 2", 
                            
                            " ",
                            html.Div(
                                children=[
                                    Select(id="dataset2",
                                        #placeholder="Dataset used for anomaly calculation",
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
                                #style={"margin-left": "5px", "vertical-align": "top"}  
                            ),
                            # " ",
                            # html.Div(
                                
                            #     #id="dataset2_container"
                            # ),
                        ),
                    ),
                    id="anomaly_period_container",
                    style={"display": "inline-block", "margin-left": "20px", "vertical-align": "top"}   
                ),
                # Esto es solo para mejorar como se trasnfieren los datos entre componentes
                dcc.Store(id="anomaly_period_values"),
                # Block("Projected Years",
                #     Number(
                #         id="start_year",
                #         default=2015,
                #         min=2015,
                #         max=2099,
                #         width="5em",
                #         debounce=False,
                #     ),
                #     "-",
                #     Number(
                #         id="end_year",
                #         default=2019,
                #         min=2015,
                #         max=2099,
                #         width="5em",
                #         debounce=False,
                #     ),
                # ),
                # Selección de años de referencia histórica
                # Block("Reference Years",
                #     Number(
                #         id="start_year_ref",
                #         default=1981,
                #         min=1951,
                #         max=2014,
                #         width="5em",
                #         debounce=False,
                #     ),
                #     "-",
                #     Number(
                #         id="end_year_ref",
                #         default=2010,
                #         min=1951,
                #         max=2014,
                #         width="5em",
                #         debounce=False,
                #     ),
                # ),
                #button_id="submit_controls",
            ),
            # Alertas para errores en el mapa
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
# Layout de controles explicativos y texto informativo
# -------------------------------------------------
from dash_extensions.javascript import arrow_function, assign
#import pandas as pd
#import numpy as np
import json 

with open("data/shapes/countries_50m.json") as f:
    countries = json.load(f)

features = [f for f in countries["features"] if f["properties"]["adm0_a3"] in ["USA", "CAN"]]
geojson_borders = {
    "type": "FeatureCollection",
    "features": features
}

#borders_pane = dl.MapPane(id="borders-pane", style={"zIndex": 300})
borders_layer = dlf.GeoJSON(
    data=geojson_borders,
    id="country-borders",
    zoomToBounds=False,
    options=dict(style=dict(color="black", weight=1.2, fill=False, fillOpacity=0)),
    interactive=True,
)


# -----------------------------
# Layers control
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

layers_control = dlf.LayersControl(
    [*[
        dlf.BaseLayer(layer, name=layer.id, checked=(i == 0))
        for i, layer in enumerate(terrenos)
    ],
    dlf.Overlay(borders_layer, name="Bordes USA/CAN",id="overlay-borders", checked=True)],
    position="topleft",
    id="layers-control"
)

# -----------------------------
# Estilo JS para geojson
# -----------------------------
style_handle = assign("""
function(feature, context){
    const {classes, colorscale, style, colorProp} = context.hideout;
    const value = feature.properties[colorProp];
    let fillColor = colorscale[0];
    for (let i = 0; i < classes.length; ++i) {
        if (value > classes[i]) fillColor = colorscale[i];
    }
    return {...style, fillColor: fillColor};
}
""")

# -----------------------------
# Info box
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
        #"position":"fixed",
        "position": "absolute",
        "top": "200px", #Y 350
        "right": "15px",
        "zIndex": 1000,
        "background": "white",
        "padding": "6px",
        "borderRadius": "4px",
        "boxShadow": "0px 0px 3px rgba(0,0,0,0.3)",
        
        # mejoras para responsividad
        "fontSize": "0.8rem",     # letra más pequeña
        "width": "fit-content",   # se ajusta al contenido
        "maxWidth": "30vw",       # no ocupa demasiado en pantallas pequeñas
        "overflow": "auto",       # aparece scroll solo si es necesario
    }
)


def controls_layout():
    return dbc.Container(
        [
            html.H5(["Harvested Weight index monitoring"]),
            html.P(
                """
                This Maproom displays projected change of key harvested weight index.
                """
            ),
            # Componente que muestra la descripción del mapa con un loading spinner
            dcc.Loading(html.P(id="map_description"), type="dot"),
            html.P(
                """
                Use the controls in the top banner to choose other variables, models,
                projected years and reference to compare with.
                """
            ),
            html.P(
                """
                Click the map (or enter coordinates) to show historical seasonal time
                series for this variable of this model, followed by a plume of
                possible projected scenarios.
                """
            ),
            dcc.Loading(html.P(id="map_description_var"), type="dot"),
            # html.P(
            #     """
            #     Change is expressed as the difference between average over projected
            #     years and average over reference historical years (in the variables
            #     units), except for precipitation and both humidity variables for
            #     which it is the relative difference (in %).
            #     """
            # ),
        ],
        fluid=True, className="scrollable-panel",
        style={"padding-bottom": "1rem", "padding-top": "1rem"},
    )


# -------------------------------------------------
# Layout del mapa principal con colorbar y marcador
# -------------------------------------------------
def map_layout():
    #print("Dentro de map layout")
    #print(store)
    #print(geojson)
    return dbc.Container(
        [
            # Título del mapa con loading
            #dcc.Loading(
            html.H5(
                id="map_title",
                style={
                    "text-align":"center", "border-width":"1px",
                    #"border-style":"solid", "border-color":"grey",
                    "margin-top":"3px", "margin-bottom":"3px",
                },
            ),#,  type="dot"),
            # Store para geojson
            #store,
            dcc.Loading(
                        dlf.Map(
                            #beforeLoad=add_draw_control,
                            children=[
                                    dlf.TileLayer(), 
                                    geojson, 
                                    #store,
                                    layers_control, 
                                    colorbar, 
                                    # dlf.FeatureGroup([
                                    #     dlf.EditControl(
                                    #         id="edit-control",
                                    #         position="bottomright",
                                    #         draw={
                                    #             "polyline": False,
                                    #             "circle": False,
                                    #             "rectangle": False,
                                    #             "polygon": False,
                                    #             "marker": True,
                                    #             "circlemarker": False,
                                    #         },
                                    #         edit={"remove": True}
                                    #     )
                                    # ], id="drawn-group")
                                    
                                    ],
                            center=[45, -95],
                            zoom=4,
                            minZoom=3,
                            maxZoom=8,
                            maxBounds=[[5, -170], [85, -10]],
                            maxBoundsViscosity=1.0,
                            style={"height": "90vh", "width": "100%"},
                            #preferCanvas=True
            ),
             type="dot", 
             #id="load",
             #className="mi-loading",
             overlay_style={"visibility":"visible", "opacity": .5, "backgroundColor": "white", #"filter": "blur(1px)"
                            }, 
             ),
            #),
             info,
            html.Div(id="dummy-output", style={"display": "none"}), 
            # Componente principal de mapa con capas y controles
            # dcc.Loading(dlf.Map(
            #     [
            #         dlf.LayersControl(id="layers_control", position="topleft"),
            #         dlf.LayerGroup(
            #             [dlf.Marker(id="loc_marker", position=(0, 0))],
            #             id="layers_group"
            #         ),
            #         dlf.ScaleControl(imperial=False, position="bottomright"),
            #         dlf.Colorbar(
            #             id="colorbar",
            #             nTicks=9,
            #             opacity=1,
            #             tooltip=True,
            #             position="topright",
            #             width=10,
            #             height=300,
            #             className="p-1",
            #             style={
            #                 "background": "white", "border-style": "inset",
            #                 "-moz-border-radius": "4px", "border-radius": "4px",
            #                 "border-color": "LightGrey",
            #             },
            #         ),
            #     ],
            #     id="map",
            #     center=None,
            #     zoom=GLOBAL_CONFIG["zoom"],
            #     style={"width": "100%", "height": "50vh"},
            # ), type="dot"),
        ],
        fluid=True,
    )


# -------------------------------------------------
# Layout para resultados, gráficos y descarga de datos
# -------------------------------------------------
def results_layout():
    return dbc.Tabs([
        dbc.Tab([
            # Botón de descarga + select alineados izquierda/derecha
            html.Div(
                dbc.Row(
                    [
                        # Izquierda: Select
                        dbc.Col(
                            html.Div(
                                [
                                    html.Span("Graph type", style={"margin-left": "5px","margin-right": "5px"}),
                                    Select(
                                        id="graph_type",
                                        #options=["bars", "lines","markers","markers-scale","lines+markers"],
                                        #labels= ["bars", "lines","dots","dots-scale","lines+dots"],
                                        options=["bars_group","bars_stack", "lines","lines+markers","markers"],
                                        labels= ["bars group","bars stack","lines","lines+dots","dots"],
                                        init=0,
                                        #style={"width": "150px"},   # mismo ancho que botón
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
                                        id="graph_scenario_container",  # <-- este es el que controlas
                                        style={"display": "flex", "align-items": "center"}
                                    ),

                                ],
                                style={"display": "flex", "align-items": "center"}
                            ),
                            width="auto"
                        ),

                        # Derecha: Botón y Download
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

            # Gráfico
            html.Div([dbc.Spinner(dcc.Graph(id="local_graph"))]),
        ], label="Local History and Projections")
    ])
