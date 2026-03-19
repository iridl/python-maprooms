from dash import html
from fieldsets import Block, PickPoint, Select
import layout_utilities as lou

from globals_ import GLOBAL_CONFIG

IRI_BLUE = "rgb(25,57,138)"
IRI_GRAY = "rgb(113,112,116)"
LIGHT_GRAY = "#eeeeee"

def app_layout():

    return lou.app_1(

        lou.navbar("Forecast",
            Block("issue_block", "Issue", html.Div(id="start_div")),
            Block("target_block", "Target Period", html.Div(id="lead_div")),
            PickPoint(width="8em"),
        ),

        lou.description(
            "Precipitation Terciles Seasonal Forecast",
            """
            The seasonal forecast for above-, below- and near-normal precipitation 
            from the IRI.
            """,
            html.P(
                """
                The default map shows globally the seasonal precipitation forecast 
                tercile probability. The historical climatology used is 1982-2010 up 
                to the forecast issued in August 2021, and is 1991-2020 from the 
                forecast issued in September 2021. The forecast shown is the latest 
                forecast made (e.g. Dec 2017) for the next season to come 
                (e.g. Jan-Mar 2018). Four different seasons are forecasted and it is 
                also possible to consult forecasts made previously. The forecasts 
                are directly computed from the extended logistic regression model as 
                tercile probabilities. 
                """
            ),
        ),
        
        lou.map(GLOBAL_CONFIG["zoom"]),
        
        lou.local_single_tabbed("Tercile Probabilities"),
    )
