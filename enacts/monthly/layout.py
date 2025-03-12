"""
The `layout.py` file is used to def a layout function, which is called within the main maproom.py file where the maproom is run.

The `layout()` function includes any code which defines the layout of the maproom. It should not include any callbacks or directly reference
the data be loaded into the maproom.
"""

# Import libraries used
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_leaflet as dlf
import xarray as xr
import json
import pandas as pd

import psycopg2
from psycopg2 import sql
import shapely
from shapely import wkb
from shapely.geometry.multipolygon import MultiPolygon
from shapely import geometry

import pingrid
import controls

from globals_ import GLOBAL_CONFIG

CONFIG = GLOBAL_CONFIG["maprooms"]["monthly"]

def layout(): # Defining the function that will be called in the layout section of  `maproom.py`.
    return dbc.Container([ # The function will return the dash bootstrap container, and all of its contents.
       dcc.Location(id="location", refresh=True),
       dbc.Row(html.H1(id="app_title")), # First of two rows (horizontal) which is the title bar of the maproom.

       dbc.Row([ # second of two rows (horizontal), which contains the rest of the maproom (the map and controls column).

           dbc.Col( # Now we divide the second row into two columns. The first column contains the controls.
               [
         # Within the controls column, we add four `Blocks` which are the controls themselves.
               controls.Block( # This block has title 'Variable' and includes a dropdown with the options listed
                 "Variable", controls.Select("variable",["Rainfall","Maximum Temperature","Minimum Temperature"])
               ),
               controls.Block( # Block containing dropdown to select from months of the year.
                  "Month", controls.Month("mon0","Jan")
               ),
           ], width=4), # End of first column; width defined here determines the width of the column.
           dbc.Col( # The second of two columns. This column contains the map.
                dbc.Container( # Container that holds the leaflet map element.
                    [ # This list holds all of the children elements within the container.
                        dlf.Map( # Dash leaflet map.
                             [
                                dlf.LayersControl(
                                    position="topleft", # Where the layers control button is placed.
                                    id="map_layers_control",
                                ),
                                dlf.LayerGroup(dlf.Marker(id="loc_marker",position=(0, 0)),id="marker_layer"),
                                dlf.ScaleControl(imperial=False, position="topright"), # Define scale bar
                                html.Div(id="map_colorbar"),
                             ],
                             id="map", # Finishing defining the dlf Map element.
                             style={ # The css style applied to the map
                                 "width": "100%",
                                 "height": "50vh",
                             },
                             center=None,
                             zoom=GLOBAL_CONFIG['zoom'],
                        ),
                        dbc.Spinner(dcc.Graph(id="plot"))
                    ],
                    fluid=True,
                    style={"padding": "0rem"},
                )
           ),
       ]),
    ])
