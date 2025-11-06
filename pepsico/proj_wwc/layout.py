from dash import html
from fieldsets import Block, Select, PickPoint, Month, Number
import layout_utilities as lou


from globals_ import GLOBAL_CONFIG

IRI_BLUE = "rgb(25,57,138)"
IRI_GRAY = "rgb(113,112,116)"
LIGHT_GRAY = "#eeeeee"


def app_layout():

    return lou.app_1(

        lou.navbar("CCA - WWC",  
            Block("Region",
                Select(
                    id="region",
                    options=["Thailand", "US-CA"],
                    labels=["Thailand", "United States and Canada"],
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
                Block("WWC statistics", Select(
                    id="variable",
                    options=[
                        "killing_frost",
                        # "warm_nights",
                        # "rain_events",
                        "dry_days",
                        "mean_Tmax",
                        "mean_Tmin",
                        "Tmax_90th-ile",
                        "Tmin_10th-ile",
                        # "heatwave_duration",
                        "frost_season_length",
                        "frost_days",
                        "wet_days",
                        # "longest_dry_spell",
                        # "longest_wet_spell",
                        "wet_day_persistence",
                        "dry_day_persistence",
                        "dry_spells_mean_length",
                        "dry_spells_median_length",
                    ],
                    labels=[
                        "Killing Frost",
                        # "Warm Nights",
                        # "Count of Rain Events",
                        "Count of Dry Days",
                        "Mean Max Temperature",
                        "Mean Min Temperature",
                        "Max Temperature 90th %-ile",
                        "Min Temperature 10th %-ile",
                        # "Mean Heatwaves Duration",
                        "Frost Season Length",
                        "Count of Frost Days",
                        "Count of Wet Days",
                        # "Longest Dry Spell",
                        # "Longest Wet Spell",
                        "Mean Wet Day Persistence",
                        "Mean Dry Day Persistence",
                        "Mean Dry Spells Length",
                        "Median Dry Spells Length",
                    ],
                    init=2,
                )),
                Block("Season",
                    Number(
                        id="start_day",
                        default=1,
                        min=1,
                        max=31,
                        width="5em",
                        debounce=False,
                    ),
                    Month(id="start_month", default="Jan"),
                    "-",
                    Number(
                        id="end_day",
                        default=31,
                        min=1,
                        max=31,
                        width="5em",
                        debounce=False,
                    ),
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
            "Weather-within-Climate Change Analysis",
            """
            This Maproom displays seasonal projected change of key
            weather-within-climate agronomic and climatic variables with respect to
            historical records.
            """
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
                units).
                """
            ),
        ),
        
        lou.map(GLOBAL_CONFIG["zoom"]),
        
        lou.local_single_tabbed(
            "Local History and Projections", download_button=True
        ),
    )
