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
                    init=1,
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
                        "warm_nights",
                        # "rain_events",
                        "mean_Tmax",
                        "mean_Tmin",
                        "dry_days",
                        "wet_days",
                        # "heatwave_duration",
                        #"frost_season_length",
                        "frost_days",
                        "Tmax_90",
                        "Tmin_10",
                        "longest_dry_spell",
                        "longest_wet_spell",
                        "wet_day_persistence",
                        "dry_day_persistence",
                        "dry_spells_mean_length",
                        "dry_spells_median_length",
                    ],
                    labels=[
                        "Warm Nights",
                        # "Count of Rain Events",
                        "Mean Max Temperature",
                        "Mean Min Temperature",
                        "Count of Dry Days",
                        "Count of Wet Days",
                        # "Mean Heatwaves Duration",
                        #"Frost Season Length",
                        "Count of Frost Days",
                        "Max Temperature 90th %-ile",
                        "Min Temperature 10th %-ile",
                        "Longest Dry Spell",
                        "Longest Wet Spell",
                        "Wet Day Persistence",
                        "Dry Day Persistence",
                        "Mean Dry Spells Length",
                        "Median Dry Spells Length",
                    ],
                    init=1,
                )),
                Block("Definitions",
                    "Frost <=",
                    Number(
                        id="frost",
                        default=-2,
                        min=-99,
                        max=0,
                        width="5em",
                        debounce=False,
                    ),
                    "˚C; ",
                    "Dry/Wet day <= / >",
                    Number(
                        id="wet",
                        default=0,
                        min=0,
                        max=999,
                        width="5em",
                        debounce=False,
                    ),
                    "mm; ",
                    "Hot/Cold day > / <=",
                    Number(
                        id="hot",
                        default=25,
                        min=-99,
                        max=999,
                        width="5em",
                        debounce=False,
                    ),
                    "˚C; ",
                    "Warm Nights Spell >=",
                    Number(
                        id="wms",
                        default=5,
                        min=0,
                        max=99,
                        width="5em",
                        debounce=False,
                    ),
                    "days ;",
                    "Dry Spell >=",
                    Number(
                        id="dryspell",
                        default=5,
                        min=0,
                        max=99,
                        width="5em",
                        debounce=False,
                    ),
                    "days",
                ),
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
                units).
                """
            ),
            html.H4(["Definitions"]),
            html.P([html.B("Mean Max/Min Temperature (mean_Tmax/min):"), """
                Average maximum/minimum temperature in season
            """]),
            html.P([html.B("Count of Dry/Wet Days (dry/wet_days):"), """
                Number of dry/wet days in the season. A dry/wet day is defined as 
                lesser or equal / greather than a user-defined threshold.
            """]),
            html.P([html.B("Count of Frost Days (frost_days):"), """
                Number of frost days in the season. A frost day is defined as 
                lesser or equal than a user-defined threshold.
            """]),
            html.P([
                html.B("Max/Min Temperature 90/10th %-ile (Tmax/min_90/10):"),"""
                    Maximum/Minimum temperature 90/10th percentile in the season. 
                    Obtained through parametric Normal distributions.
                """
            ]),
            html.P([
                html.B("Warm Nights (warm_nights):"),"""
                    Number of days in a season in a warm night spell. A warm night 
                    spell is defined as a user-defined minimum consecutive number of 
                    warm nights. A warm night is defined as days where minimum 
                    temperature is greater than a user-defined thredhold.
                """
            ]),
            html.P([
                html.B("Longest Dry/Wet Spell (longest_dry/wet_spell):"),"""
                    Length of longest dry/wet spell in the season. A dry/wet spell 
                    is defined as consecutive dry/wet days. A dry/wet day is defined 
                    as lesser or equal / greather than a user-defined threshold.
                """
            ]),
            html.P([
                html.B("Dry/Wet Day Persistence (dry/wet_day_persistence):"),"""
                    Ratio of cumulative (at least 2) dry/wet days against total 
                    dry/wet days in the season. A dry/wet day is defined as lesser 
                    or equal / greather than a user-defined threshold.
                """
            ]),
            html.P([
                html.B(
                    "Mean/Median Dry Spells Length (dry_spells_mean/median_length):"
                ),"""
                    Mean/Median of dry spells length in the season. A dry spell is 
                    defined as a user-defined minimum consecutive number of dry 
                    days. A dray day is defined as days where precipitation is 
                    lesser or equal than a user-defined thredhold.
                """
            ]),
        ),
        
        lou.map(GLOBAL_CONFIG["zoom"]),
        
        lou.local_single_tabbed(
            "Local History and Projections", download_button=True
        ),
    )
