from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_leaflet as dlf
from fieldsets import Block, Select, PickPoint, Month, Number
import layout_utilities as lou

from globals_ import GLOBAL_CONFIG

IRI_BLUE = "rgb(25,57,138)"
IRI_GRAY = "rgb(113,112,116)"
LIGHT_GRAY = "#eeeeee"


def app_layout():

    return lou.app_1(

        lou.navbar("Forecast",
            Block("proba_block", "Probability", Select(
                id="proba",
                options=["exceeding", "non-exceeding"],
            )),
            Block("var_block", "Variable", Select(
                id="variable",
                options=["Percentile", "Value"],
            )),
            Block("perc_block", "Percentile", Select(
                id="percentile",
                options=range(10, 95, 5),
                init=8,
            ), " %-ile"),
            Block("thresh_block", "Threshold",
                Number(id="threshold", default=0, debounce=False), 
                html.Div(id='phys-units', style={"color": "white"}),
            ),
            Block("issue_block", "Issue", html.Div(id="start_div")),
            Block("target_block", "Target Period", html.Div(id="lead_div")),
            PickPoint(width="8em"),
        ),

        lou.description(
            "Forecast",
            """
            This Maproom displays the full forecast distribution 
            in different flavors.
            """,
            html.P(
                """
                The map shows the probability of exceeding or non-exceeding
                an observed historical percentile or a threshold in the variable physical units
                for a given forecast (issue date and target period).
                Use the controls in the top banner to choose presentation of the forecast to map
                and to navigate through other forecast issues and targets.
                """
            ),
            html.P(
                """
                Click the map to show forecast and observed
                probability of exceeding and distribution
                at the clicked location.
                """
            ),
        ),
        
        lou.map(GLOBAL_CONFIG["zoom"]),
        
        lou.local_double_tabbed(
            ["Probability of Exceedance", "Probability Distribution"],
        ),
    )
