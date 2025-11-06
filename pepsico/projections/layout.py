from dash import html
from fieldsets import Block, Select, PickPoint, Month, Number
import layout_utilities as lou


from globals_ import GLOBAL_CONFIG

IRI_BLUE = "rgb(25,57,138)"
IRI_GRAY = "rgb(113,112,116)"
LIGHT_GRAY = "#eeeeee"


def app_layout():

    return lou.app_1(

        lou.navbar("CCA",  
            Block("Region",
                Select(
                    id="region",
                    options=["SAMER", "SASIA", "Thailand", "US-CA"],
                    labels=[
                        "South America",
                        "South Asia",
                        "Thailand",
                        "United States and Canada",
                    ],
                    init=3,
                ),
            ),
            PickPoint(width="8em"),
            Block("Submit",
                Block("Scenario", Select(
                    id="scenario",
                    options=["picontrol", "ssp126", "ssp370", "ssp585"],
                    init=1,
                )),
                Block("Model", Select(id="model", options=[
                    "Multi-Model-Average", "GFDL-ESM4", "IPSL-CM6A-LR",
                    "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL",
                ])),
                Block("Variable", Select(
                    id="variable",
                    options=[
                        "hurs",
                        "huss",
                        "pr",
                        "prsn",
                        "ps",
                        "rlds",
                        "rsds",
                        "sfcwind",
                        "tas",
                        "tasmax",
                        "tasmin",
                    ],
                    labels=[
                        "Near-Surface Relative Humidity",
                        "Near-Surface Specific Humidity",
                        "Precipitation",
                        "Snowfall Flux",
                        "Surface Air Pressure",
                        "Surface Downwelling Longwave Radiation",
                        "Surface Downwelling Shortwave Radiation",
                        "Near-Surface Wind Speed",
                        "Near-Surface Air Temperature",
                        "Daily Maximum Near-Surface Air Temperature",
                        "Daily Minimum Near-Surface Air Temperature",
                    ],
                    init=2,
                )),
                Block("Season",
                    Month(id="start_month", default="Jan"),
                    "-",
                    Month(id="end_month", default="Mar"),
                ),
                Block("Projected Years",
                    Number(
                        id="start_year",
                        default=2015,
                        min=2015,
                        max=2099,
                        width="5em",
                        debounce=False,
                    ),
                    "-",
                    Number(
                        id="end_year",
                        default=2019,
                        min=2015,
                        max=2099,
                        width="5em",
                        debounce=False,
                    ),
                ),
                Block("Reference Years",
                    Number(
                        id="start_year_ref",
                        default=1981,
                        min=1951,
                        max=2014,
                        width="5em",
                        debounce=False,
                    ),
                    "-",
                    Number(
                        id="end_year_ref",
                        default=2010,
                        min=1951,
                        max=2014,
                        width="5em",
                        debounce=False,
                    ),
                ),
                button_id="submit_controls",
            ),
        ),

        lou.description(
            "Climate Change Analysis",
            """
            This Maproom displays seasonal projected change of key climate
            variables with respect to historical records.
            """,
            html.P(
                """
                Use the controls in the top banner to choose other variables, models,
                scenarios, seasons, projected years and reference to compare with.
                """
            ),
            html.P(
                """
                Click the map (or enter coordinates) to show historical seasonal time
                series for this variable of this model, followed by a plume of
                possible projected scenarios.
                """
            ),
            html.P(
                """
                Change is expressed as the difference between average over projected
                years and average over reference historical years (in the variables
                units), except for precipitation and both humidity variables for
                which it is the relative difference (in %).
                """
            ),
        ),
        
        lou.map(GLOBAL_CONFIG["zoom"]),
        
        lou.local_single_tabbed(
            "Local History and Projections", download_button=True
        ),
    )
