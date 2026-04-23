from dash import html
from fieldsets import Block, PickPoint
import layout_utilities as lou

from globals_ import GLOBAL_CONFIG

IRI_BLUE = "rgb(25,57,138)"
IRI_GRAY = "rgb(113,112,116)"
LIGHT_GRAY = "#eeeeee"

def app_layout(config):

    return lou.app_1(

        lou.navbar("Forecast",
            Block("issue_block", "Issue", html.Div(id="start_div")),
            Block("target_block", "Target Period", html.Div(id="lead_div")),
            PickPoint(width="8em"),
        ),

        lou.description(
            f"{config['variable']} Terciles {config['forecast']} Forecast",
            f"""
            The {config['forecast']} forecast for above-, below- and near-normal 
            {config['variable']} from the IRI.
            """,
            html.P(
                f"""
                The default map shows globally the {config['forecast']} 
                {config['variable']} forecast tercile probability. The historical 
                climatology used is {config['clim']}. The forecast shown is the 
                latest forecast made for the next season to come. The forecasts are 
                directly computed from the extended logistic regression model as 
                tercile probabilities. 
                """
            ),
        ),
        
        lou.map(GLOBAL_CONFIG["zoom"], scale_control_position="bottomleft", colorbars={
            "below_cb" : {
                "nTicks": 6, "min": 37.5, "max": 100, "tickDecimals": 1, "unit": "%",
                "opacity": 1, "tooltip": True, "position": "bottomright",
                "width": 10, "height": 140,
            },
            "above_cb" : {
                "nTicks": 6, "min": 37.5, "max": 100, "tickDecimals": 1, "unit": "%",
                "opacity": 1, "tooltip": True, "position": "topright",
                "width": 10, "height": 140,
            },
        }),
        
        lou.local_single_tabbed("Tercile Probabilities"),
    )
