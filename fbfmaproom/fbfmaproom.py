import cftime
from typing import Any, Dict, Tuple, Optional
import os
import threading
import time
import io
import datetime
import urllib.parse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr
import cv2
import flask
import dash
from dash import html
from dash.dependencies import Output, Input, State, ALL
from dash.exceptions import PreventUpdate
import shapely
from shapely import wkb
from shapely.geometry import Polygon, Point
from shapely.geometry.multipoint import MultiPoint
import psycopg2
from psycopg2 import sql
import math
import traceback
import enum
import itertools
import uuid
import warnings
import yaml

import __about__ as about
import pingrid
from pingrid import ClientSideError, CMAPS, InvalidRequestError, NotFoundError, parse_arg
from pingrid import Color
import fbflayout
import fbftable
import dash_bootstrap_components as dbc

from collections import OrderedDict


SERVER = flask.Flask(__name__)

SERVER.register_error_handler(ClientSideError, pingrid.client_side_error)

month_abbrev = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
abbrev_to_month0 = dict((abbrev, month0) for month0, abbrev in enumerate(month_abbrev))


class FbfDash(dash.Dash):
    def index(self, *args, **kwargs):
        path = kwargs['path']
        if not is_valid_root(path):
            raise NotFoundError(f"Unknown resource {path}")
        return super().index(*args, **kwargs)


def is_valid_root(path):
    if path in CONFIG["countries"]:
        return True
    return False


def fill_config(config):
    """Replaces parts of the config dictionary with objects"""
    for country in config["countries"].values():
        for key, dataset in country["datasets"]["observations"].items():
            country["datasets"]["observations"][key] = ObsDataset(**dataset)
        for key, dataset in country["datasets"]["forecasts"].items():
            country["datasets"]["forecasts"][key] = ForecastDataset(**dataset)


DEFAULT = object()

class Dataset:
    def __init__(
            self, *, label, description=None, units=DEFAULT, path,
            var_names, colormap, format='number1', range=None
    ):
        # TODO: some datasets we pulled from ingrid already have colormap,
        # scale_max, and scale_min attributes. Should we just use those,
        # instead of getting them from the config file and/or computing
        # them?
        self.label = label
        self.description = description
        self._units = units
        self.path = path
        self.var_names = var_names
        self.colormap = CMAPS[colormap]
        self.format = format_funcs[format]

        assert range is None or len(range) == 2
        self.range = range


    @property
    def units(self):
        # If config file doesn't specify units then read from the
        # dataset, and then cache the result so we don't have to read
        # it again.
        if self._units is DEFAULT:
            self._units = self.open().attrs.get('units')
        return self._units

    def open(self, val_min=None, val_max=None):
        path = data_path(self.path)
        try:
            ds = xr.open_zarr(path, consolidated=False)
        except Exception as e:
            raise Exception(f"Couldn't open {path}") from e
        ds = ds.rename({
            v: k
            for k, v in self.var_names.items()
            if v is not None and v != k
        })
        da = ds["value"]


        if val_min is None:
            val_min = self.range[0]
        if val_max is None:
            val_max = self.range[1]
        da.attrs["colormap"] = self.colormap
        da.attrs["scale_min"] = val_min
        da.attrs["scale_max"] = val_max
        return da


class ObsDataset(Dataset):
    def __init__(self, *, lower_is_worse, **kwargs):
        super().__init__(**kwargs)
        self.lower_is_worse = lower_is_worse

    def open(self, val_min=0.0, val_max=1000.0):
        da = super().open(val_min, val_max)
        if da.dtype == 'timedelta64[ns]':
            da = (da / np.timedelta64(1, 'D')).astype(float)
        return da

    def select(self, target_month0, target_year=None):
        da = self.open()
        if target_year is not None:
            target_date = (
                cftime.Datetime360Day(target_year, 1, 1) +
                pd.Timedelta(target_month0 * 30, unit='days')
            )
            da = da.sel(time=target_date)

        with warnings.catch_warnings():
            # ds.where in xarray 2022.3.0 uses deprecated numpy
            # functionality. A recent change deletes the offending line;
            # see if this catch_warnings can be removed once that's
            # released.
            # https://github.com/pydata/xarray/commit/3a320724100ab05531d8d18ca8cb279a8e4f5c7f
            warnings.filterwarnings("ignore", category=DeprecationWarning, module='numpy.core.fromnumeric')
            da = da.where(lambda x: x["time"].dt.month == target_month0 + 0.5, drop=True)

        return da


class ForecastDataset(Dataset):
    # forecasts are always expressed as the probability of
    # something bad happening
    lower_is_worse = False

    def __init__(self, *, is_poe, **kwargs):
        super().__init__(**kwargs)
        self.is_poe = is_poe

    def open(self, val_min=0.0, val_max=100.0):
        return super().open(val_min, val_max)

    def select(self, issue_month0, target_month0, target_year=None, freq=None):
        l = (target_month0 - issue_month0) % 12

        da = self.open()

        issue_dates = da["issue"].where(da["issue"].dt.month == issue_month0 + 1, drop=True)
        da = da.sel(issue=issue_dates)

        # Now that we have only one issue month, each target date uniquely
        # identifies a single forecast, so we can replace the issue date
        # coordinate with a target_date coordinate.
        l_delta = pd.Timedelta(l * 30, unit='days')
        da = da.assign_coords(
            target_date=("issue", (da["issue"] + l_delta).data)
        ).swap_dims({"issue": "target_date"}).drop_vars("issue")

        if "lead" in da.coords:
            da = da.sel(lead=l)

        if target_year is not None:
            target_date = (
                cftime.Datetime360Day(target_year, 1, 1) +
                pd.Timedelta(target_month0 * 30, unit='days')
            )
            try:
                da = da.sel(target_date=target_date)
            except KeyError:
                raise NotFoundError(f'No forecast for issue_month0 {issue_month0} in year {target_year}') from None

        if freq is not None:
            if self.is_poe:
                # Forecasts are always expressed as the probability of a
                # bad year, so probability of exceedance for variables for
                # which higher is worse, and probability of non-exceedance
                # for variables for which lower is worse. When the slider
                # is set to n, it means to select the n% worst years, so
                # if it's a probability of exceedance we select the
                # probability of exceeding the (100-n)th percentile, and
                # if it's a probability of non-exceedance we select the
                # probability of not exceeding the nth percentile.
                percentile = 100 - freq
            else:
                percentile = freq
            da = da.sel(pct=percentile, drop=True)

        return da




def table_columns(dataset_config, predictor_keys, predictand_key,
                  season_length):
    tcs = OrderedDict()
    tcs["time"] = dict(
        name="Year",
        format=lambda midpoint: year_label(midpoint, season_length),
        tooltip=None,
        type=ColType.SPECIAL,
    )
    for key in predictor_keys + [predictand_key]:
        tcs[key] = make_column(key, dataset_config)

    return tcs


def make_column(key, dataset_config):
    if key in dataset_config['forecasts']:
        col_type = ColType.FORECAST
        ds_config = dataset_config['forecasts'][key]
    elif key in dataset_config['observations']:
        col_type = ColType.OBS
        ds_config = dataset_config['observations'][key]
    else:
        assert False, f'Unknown dataset key {key}'

    return dict(
        name=ds_config.label,
        units=ds_config.units,
        format=ds_config.format,
        tooltip=ds_config.description,
        lower_is_worse=ds_config.lower_is_worse,
        type=col_type,
    )

class ColType(enum.Enum):
    FORECAST = enum.auto()
    OBS = enum.auto()
    SPECIAL = enum.auto()


def number_formatter(precision):
    def f(x):
        if np.isnan(x):
            return ""
        return f"{x:.{precision}f}"
    return f


def format_bad(x):
    if np.isnan(x) or np.isclose(x, 0):
        return ""
    else:
        return "Bad"


def format_timedelta_days(x):
    return number_formatter(2)(x.days + x.seconds / 60 / 60 / 24)


def format_enso(x):
    if np.isnan(x):
        return ""
    if np.isclose(x, 1):
        return "La Niña"
    if np.isclose(x, 2):
        return "Neutral"
    if np.isclose(x, 3):
        return "El Niño"
    assert False, f"Unknown enso state {x}"


format_funcs = {
    'number0': number_formatter(0),
    'number1': number_formatter(1),
    'number2': number_formatter(2),
    'number3': number_formatter(3),
    'number4': number_formatter(4),
    'timedelta_days': format_timedelta_days,
    'bad': format_bad,
    'enso': format_enso,
}

CONFIG = pingrid.load_config(os.environ["CONFIG"])
fill_config(CONFIG)

PFX = CONFIG["core_path"]
TILE_PFX = CONFIG["tile_path"]
ADMIN_PFX = CONFIG["admin_path"]

APP = FbfDash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    server=SERVER,
    url_base_pathname=f"{PFX}/",
    meta_tags=[
        {"name": "description", "content": "content description 1234"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)
APP.title = "FBF--Maproom"

APP.layout = fbflayout.app_layout()


def data_path(relpath):
    return Path(CONFIG["data_root"], relpath)


def from_month_since_360Day(months):
    year = 1960 + months // 12
    month_zero_based = math.floor(months % 12)
    day_zero_based = ((months % 12) - month_zero_based) * 30
    return cftime.Datetime360Day(year, month_zero_based + 1, day_zero_based + 1)


def year_label(midpoint, season_length):
    half_season = datetime.timedelta(season_length / 2 * 30)
    start = midpoint - half_season
    end = midpoint + half_season - datetime.timedelta(days=1)
    if start.year == end.year:
        label = str(start.year)
    else:
        label = f"{start.year}/{end.year % 100}"
    return label


def geometry_containing_point(
    country_key: str, point: Tuple[float, float], mode: str
):
    df = retrieve_vulnerability(country_key, mode, 2020)  # arbitrary year
    x, y = point
    p = Point(x, y)
    geom, attrs = None, None
    for _, r in df.iterrows():
        minx, miny, maxx, maxy = r["the_geom"].bounds
        if minx <= x <= maxx and miny <= y <= maxy and r["the_geom"].contains(p):
            geom = r["the_geom"]
            attrs = {k: v for k, v in r.items() if k not in ("the_geom")}
            break
    return geom, attrs


def retrieve_vulnerability(
    country_key: str, mode: str, year: int
) -> pd.DataFrame:
    config = CONFIG["countries"][country_key]
    sc = config["shapes"][int(mode)]
    vuln_sql = sc.get(
        "vuln_sql",
        "select cast(null as int) as key, 0 as year, 0 as vuln where 1 = 2"
    )
    with psycopg2.connect(**CONFIG["db"]) as conn:
        s = sql.Composed(
            [
                sql.SQL("with v as ("),
                sql.SQL(vuln_sql),
                sql.SQL("), g as ("),
                sql.SQL(sc["sql"]),
                sql.SQL(
                    """
                    ), a as (
                        select
                            key,
                            avg(vuln) as mean,
                            stddev_pop(vuln) as stddev
                        from v
                        group by key
                    )
                    select
                        g.label, g.key, g.the_geom,
                        v.year,
                        v.vuln as vulnerability,
                        a.mean as mean,
                        a.stddev as stddev,
                        v.vuln / a.mean as normalized,
                        coalesce(to_char(v.vuln,'999,999,999,999'),'N/A') as "Vulnerability",
                        coalesce(to_char(a.mean,'999,999,999,999'),'N/A') as "Mean",
                        coalesce(to_char(a.stddev,'999,999,999,999'),'N/A') as "Stddev",
                        coalesce(to_char(v.vuln / a.mean,'999,990D999'),'N/A') as "Normalized"
                    from (g left outer join a using (key))
                        left outer join v on(g.key=v.key and v.year=%(year)s)
                    """
                ),
            ]
        )
        # print(s.as_string(conn))
        df = pd.read_sql(
            s,
            conn,
            params=dict(year=year),
        )
    # print("bytes: ", sum(df.the_geom.apply(lambda x: len(x.tobytes()))))
    df["the_geom"] = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))
    return df


def generate_tables(
    country_key,
    season_config,
    table_columns,
    predictand_key,
    issue_month0,
    freq,
    mode,
    geom_key,
    final_season,
):

    basic_ds = fundamental_table_data(
        country_key, table_columns, season_config, issue_month0,
        freq, mode, geom_key
    )
    if "pct" in basic_ds.coords:
        basic_ds = basic_ds.drop_vars("pct")
    basic_df = basic_ds.to_dataframe()
    main_df, summary_df, thresholds = augment_table_data(
        basic_df, freq, table_columns, predictand_key, final_season
    )
    return main_df, summary_df, thresholds


def region_shape(mode, country_key, geom_key):
    if mode == "pixel":
        [[y0, x0], [y1, x1]] = json.loads(geom_key)
        shape = Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
    else:
        config = CONFIG["countries"][country_key]
        base_query = config["shapes"][int(mode)]["sql"]
        response = subquery_unique(base_query, geom_key, "the_geom")
        shape = wkb.loads(response.tobytes())
    return shape


def region_label(country_key: str, mode: int, region_key: str):
    if mode == "pixel":
        label = None
    else:
        config = CONFIG["countries"][country_key]
        base_query = config["shapes"][int(mode)]["sql"]
        label = subquery_unique(base_query, region_key, "label")
    return label


def subquery_unique(base_query, key, field):
    query = sql.Composed(
        [
            sql.SQL(
                "with a as (",
            ),
            sql.SQL(base_query),
            sql.SQL(
                ") select {} from a where key::text = %(key)s"
            ).format(sql.Identifier(field)),
        ]
    )
    with psycopg2.connect(**CONFIG["db"]) as conn:
        df = pd.read_sql(query, conn, params={"key": key})
    if len(df) == 0:
        raise InvalidRequestError(f"invalid region {key}")
    assert len(df) == 1
    return df.iloc[0][field]


def fundamental_table_data(country_key, table_columns,
                           season_config, issue_month0, freq, mode,
                           geom_key):
    year_min = season_config["start_year"]
    season_length = season_config["length"]
    target_month0 = season_config["target_month"]
    datasets = CONFIG["countries"][country_key]["datasets"]

    forecast_ds = xr.Dataset(
        data_vars={
            forecast_key: datasets["forecasts"][forecast_key].select(
                issue_month0, target_month0, freq=freq
            ).rename({'target_date':"time"})
            for forecast_key, col in table_columns.items()
            if col["type"] is ColType.FORECAST
        }
    )
    shape = region_shape(mode, country_key, geom_key)

    forecast_ds = value_for_geom(forecast_ds, country_key, geom_key, shape)

    obs_das = [
        datasets["observations"][col_key].select(target_month0).rename(col_key)
        for col_key, col in table_columns.items()
        if col["type"] is ColType.OBS
    ]
    obs_ds = xr.merge(
        [
            value_for_geom(da, country_key, geom_key, shape)
            for da in obs_das
        ]
    )

    main_ds = xr.merge(
        [
            forecast_ds,
            obs_ds,
        ]
    )

    year = main_ds["time"].dt.year
    main_ds = main_ds.where(year >= year_min, drop=True)

    main_ds = main_ds.sortby("time", ascending=False)

    return main_ds


def value_for_geom(ds, country_key, geom_key, shape):
    if 'lon' in ds.coords:
        result = pingrid.average_over(ds, shape, all_touched=True)
    elif 'geom_key' in ds.coords:
        if geom_key in ds['geom_key']:
            result = ds.sel(geom_key=geom_key)
        else:
            # TODO: use geopandas intersection
            # - fetch all the shapes whose keys are values of ds.geom_key
            #   and that intersect the target shape
            # - calculate areas of intersection
            # - calculate average value weighted by area of intersection
            raise Exception("Not implemented")
    else:
        # ds has no spatial dimension; return it as-is.
        result = ds

    return result


def augment_table_data(main_df, freq, table_columns, predictand_key, final_season):
    main_df = main_df.copy()

    main_df["time"] = main_df.index.to_series()

    regular_keys = [
        key for key, col in table_columns.items()
        if col["type"] is not ColType.SPECIAL
    ]
    regular_data = {
        key: main_df[key].dropna()
        for key in regular_keys
    }

    def is_ascending(col_key):
        return table_columns[col_key]["lower_is_worse"]

    def percentiles(df, is_ascending):
        if final_season is not None:
            df = df.where(lambda x: x.index <= final_season, np.nan)
        df = df.rank(method="min", ascending=is_ascending, pct=True)
        return df

    rank_pct = {
        key: percentiles(regular_data[key], is_ascending(key))
        for key in regular_keys
    }

    worst_flags = {}
    thresholds = {}
    for key in regular_keys:
        vals = regular_data[key]
        if len(vals.unique()) <= 3:
            # special case for legacy boolean bad years
            if is_ascending(key):
                bad_val = vals.min()
            else:
                bad_val = vals.max()
            worst_flags[key] = vals == bad_val
        else:
            worst_flags[key] = (rank_pct[key] <= freq / 100).astype(bool)
        worst_vals = regular_data[key][worst_flags[key]]
        if is_ascending(key):
            thresholds[key] = worst_vals.max()
        else:
            thresholds[key] = worst_vals.min()

    bad_year = worst_flags[predictand_key].dropna().astype(bool)

    summary_df = pd.DataFrame()
    for key in regular_keys:
        if key != predictand_key:
            summary_df[key] = hits_and_misses(worst_flags[key], bad_year)
        main_df[f"worst_{key}"] = worst_flags[key].astype(int)

    return main_df, summary_df, thresholds


def format_ganttit(
        variable,
        var_name,
        format_thresh,
        lower_is_worse,
        thresh,
        country,
        mode,
        freq,
        season_id,
        issue_month0,
        geom_key,
        region_label,
        severity,
):
    if mode == 'pixel':
        bounds = geom_key
        region_id = ''
        assert region_label is None
        region_label = ''
    else:
        bounds = ''
        region_id = geom_key

    id_ = str(uuid.uuid4())
    season_config = CONFIG['countries'][country]['seasons'][season_id]
    args = {
        'variable': variable,
        'country': country,
        'mode': mode,
        'freq': freq,
        'thresh': float(thresh),
        'season': {
            'id': season_id,
            'label': season_config['label'],
            'target_month': season_config['target_month'],
            'length': season_config['length'],
        },
        'issue_month': issue_month0,
        'bounds': bounds,
        'region': {
            'id': region_id,
            'label': region_label,
        },
        'severity': severity,
    }
    url = CONFIG["gantt_url"] + urllib.parse.urlencode(dict(data=json.dumps(args)))
    more_less = "less" if lower_is_worse else "greater"
    component = html.A(
        [
            dbc.Button(
                "Set trigger", id=id_, size='sm'
            ),
            dbc.Tooltip(
                f"Click to trigger if {var_name} is {format_thresh(thresh)} or {more_less}",
                target=id_,
                className="tooltiptext",
            ),
        ],
        href=url,
        target="_blank",
    )
    return component


def format_summary_table(summary_df, table_columns, thresholds,
                         country, mode, freq, season_id, issue_month0,
                         geom_key, severity
):
    format_accuracy = lambda x: f"{x * 100:.2f}%"
    format_count = lambda x: f"{x:.0f}"

    formatted_df = pd.DataFrame()

    formatted_df["time"] = [
        fbftable.head_cell(text, tooltip)
        for text, tooltip in (
            ("", None),
            ("Worthy-action:", "Drought was forecasted and a ‘bad year’ occurred"),
            ("Act-in-vain:", "Drought was forecasted but a ‘bad year’ did not occur"),
            ("Fail-to-act:", "No drought was forecasted but a ‘bad year’ occurred"),
            ("Worthy-Inaction:", "No drought was forecasted, and no ‘bad year’ occurred"),
            ("Rate:", "Percentage of worthy-action and worthy-inactions"),
            ("Threshold:", "Threshold for a forecast of drought"),
        )
    ]

    label = region_label(country, mode, geom_key)
    for c in summary_df.columns:
        formatted_df[c] = (
            [format_ganttit(
                c,
                table_columns[c]['name'],
                table_columns[c]['format'],
                table_columns[c]['lower_is_worse'],
                thresholds[c],
                country,
                mode,
                freq,
                season_id,
                issue_month0,
                geom_key,
                label,
                severity,
            )] +
            list(map(format_count, summary_df[c][0:4])) +
            [
                format_accuracy(summary_df[c][4]),
                table_columns[c]['format'](thresholds[c])
            ]
        )

    for c in set(table_columns) - set(formatted_df.columns):
        formatted_df[c] = ''

    return formatted_df


def hits_and_misses(prediction, truth):
    assert pd.notnull(prediction).all()
    assert pd.notnull(truth).all()
    true_pos = (prediction & truth).sum()
    false_pos = (prediction & ~truth).sum()
    false_neg = (~prediction & truth).sum()
    true_neg = (~prediction & ~truth).sum()
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    return [true_pos, false_pos, false_neg, true_neg, accuracy]


def calculate_bounds(pt, res, origin):
    x, y = pt
    dx, dy = res
    x0, y0 = origin
    cx = (x - x0 + dx / 2) // dx * dx + x0
    cy = (y - y0 + dy / 2) // dy * dy + y0
    return [[cx - dx / 2, cy - dy / 2], [cx + dx / 2, cy + dy / 2]]


def country(pathname: str) -> str:
    return pathname.split("/")[2]


APP.clientside_callback(
    """
    function update_layout(disp) {
        var lclass = ""
        var rclass = ""

        if (!disp.includes("Map")) {
            lclass = "d-none"
        }
        if (!disp.includes("Table")) {
            rclass = "d-none"
        }
        return [ lclass, rclass ]
    }
    """,
    Output("lcol", "className"),
    Output("rcol", "className"),
    Input("fbf_display", "value"),
)


@APP.callback(
    Output("logo", "src"),
    Output("map", "center"),
    Output("map", "zoom"),
    Output("season", "options"),
    Output("season", "value"),
    Output("vuln_colorbar", "colorscale"),
    Output("mode", "options"),
    Output("mode", "value"),
    Output("predictand", "options"),
    Output("predictand", "value"),
    Output("predictors", "options"),
    Output("predictors", "value"),
    Output("freq", "value"),
    Output("severity", "value"),
    Output("include_upcoming", "value"),
    Output("modal", "is_open"),
    Output("modal-body", "children"),
    Input("location", "pathname"),
    State("location", "search"),
)
def initial_setup(pathname, qstring):
    country_key = country(pathname)
    c = CONFIG["countries"][country_key]

    season_options = [
        dict(
            label=c["seasons"][k]["label"],
            value=k,
        )
        for k in sorted(c["seasons"].keys())
    ]
    cx, cy = c["center"]
    vuln_cs = CMAPS[c["datasets"]["vuln"]["colormap"]].to_dash_leaflet()
    mode_options = [
        dict(
            label=k["name"],
            value=str(i),
        )
        for i, k in enumerate(c["shapes"])
    ] + [dict(label="Pixel", value="pixel")]

    datasets_config = c["datasets"]
    predictors_options = predictand_options = [
        dict(
            label=v.label,
            value=k,
        )
        for k, v in itertools.chain(
            datasets_config["forecasts"].items(),
            datasets_config["observations"].items()
        )
    ]

    mode_value = parse_arg("mode", default="0", qstring=qstring)
    season_value = parse_arg(
        "season", default=min(c["seasons"].keys()), qstring=qstring
    )

    predictors_value = parse_arg(
        "predictors",
        conversion=lambda x: x.split(" "),
        default=datasets_config["defaults"]["predictors"],
        qstring=qstring
    )

    predictand_value = parse_arg(
        "predictand",
        default=datasets_config["defaults"]["predictand"],
        qstring=qstring
    )

    freq_value = parse_arg(
        "freq",
        conversion=int,
        default=30,
        qstring=qstring
    )

    severity_value = parse_arg(
        "severity",
        conversion=int,
        default=0,
        qstring=qstring
    )

    include_upcoming_value = parse_arg(
        "include_upcoming",
        conversion=bool,
        default=False,
        qstring=qstring
    )

    warning = c.get("onload_warning")
    show_modal = (
        warning is not None and
        parse_arg(
            "show_modal",
            conversion=json.loads,
            default=True,
            qstring=qstring
        )
    )


    return (
        f"{PFX}/custom/{c['logo']}",
        [cy, cx],
        c["zoom"],
        season_options,
        season_value,
        vuln_cs,
        mode_options,
        mode_value,
        predictand_options,
        predictand_value,
        predictors_options,
        predictors_value,
        freq_value,
        severity_value,
        include_upcoming_value,
        show_modal,
        warning,
    )

@SERVER.route(f"{PFX}/custom/<path:relpath>")
def custom_static(relpath):
    return flask.send_from_directory(CONFIG["custom_asset_path"], relpath)

@APP.callback(
    Output("year", "options"),
    Output("year", "value"),
    Output("issue_month", "options"),
    Output("issue_month", "value"),
    Input("season", "value"),
    Input("map_column", "value"),
    Input("location", "pathname"),
    State("location", "search"),
)
def forecast_selectors(season, col_name, pathname, qstring):
    country_key = country(pathname)
    country_conf = CONFIG["countries"][country_key]
    season_conf = country_conf["seasons"][season]

    year_min = season_conf["start_year"]
    fcst = country_conf["datasets"]["forecasts"][col_name].open()
    latest_issue = fcst["issue"].max().item()
    if season_conf["target_month"] < latest_issue.month:
        year_max = latest_issue.year + 1
    else:
        year_max = latest_issue.year
    year_range = range(year_max, year_min - 1, -1)

    midpoints = [
        cftime.Datetime360Day(year, 1, 1) + pd.Timedelta(days=season_conf["target_month"] * 30)
        for year in year_range
    ]
    year_options = [
        dict(
            label=year_label(midpoint, season_conf["length"]),
            value=midpoint.year
        )
        for midpoint in midpoints
    ]
    issue_month_options = [
        dict(
            label=pd.to_datetime(v + 1, format="%m").month_name(),
            value=month_abbrev[v],
        )
        for v in reversed(season_conf["issue_months"])
    ]

    year_value = parse_arg(
        "year",
        conversion=int,
        default=year_max,
        qstring=qstring
    )
    issue_month_value = parse_arg(
        "issue_month",
        default=month_abbrev[season_conf["issue_months"][-1]],
        qstring=qstring
    )

    return (
        year_options,
        year_value,
        issue_month_options,
        issue_month_value,
    )


@APP.callback(
    Output("marker", "position"),
    Input("location", "pathname"),
    Input("map", "click_lat_lng"),
    State("location", "search"),
)
def map_click(pathname, lat_lng, qstring):
    if lat_lng is None:  # initial call at page load
        country_key = country(pathname)
        x, y = CONFIG["countries"][country_key]["marker"]
        default = (y, x)
        result = parse_arg("position", conversion=json.loads, default=default, qstring=qstring)
    else:
        result = lat_lng
    return result


@APP.callback(
    Output("outline", "data"),
    Output("geom_key", "data"),
    Input("marker", "position"),
    Input("mode", "value"),
    State("location", "pathname"),
)
def update_selected_region(position, mode, pathname):
    country_key = country(pathname)
    y, x = position
    c = CONFIG["countries"][country_key]
    selected_shape = None
    key = None
    if mode == "pixel":
        (x0, y0), (x1, y1) = calculate_bounds(
            (x, y), c["resolution"], c.get("origin", (0, 0))
        )
        pixel = box(x0, y0, x1, y1)
        geom, _ = geometry_containing_point(country_key, tuple(c["marker"]), "0")
        if pixel.intersects(geom):
            selected_shape = box(x0, y0, x1, y1)
        key = str([[y0, x0], [y1, x1]])
    else:
        geom, attrs = geometry_containing_point(country_key, (x, y), mode)
        if geom is not None:
            selected_shape = geom
            key = str(attrs["key"])
    if selected_shape is None:
        selected_shape = ZERO_SHAPE

    geojson = shapely.geometry.mapping(selected_shape)
    return {'features': [geojson]}, key


def box(x0, y0, x1, y1):
    return MultiPoint([(x0, y0), (x1, y1)]).envelope


ZERO_SHAPE = box(0, 0, 0, 0)


@APP.callback(
    Output("marker_popup", "children"),
    Input("location", "pathname"),
    Input("marker", "position"),
    Input("mode", "value"),
)
def update_popup(pathname, position, mode):
    country_key = country(pathname)
    y, x = position
    c = CONFIG["countries"][country_key]
    title = "No Data"
    content = []
    if mode == "pixel":
        (x0, y0), (x1, y1) = calculate_bounds(
            (x, y), c["resolution"], c.get("origin", (0, 0))
        )
        pixel = box(x0, y0, x1, y1)
        geom, _ = geometry_containing_point(country_key, tuple(c["marker"]), "0")
        if pixel.intersects(geom):
            px = (x0 + x1) / 2
            pxs = "E" if px > 0.0 else "W" if px < 0.0 else ""
            py = (y0 + y1) / 2
            pys = "N" if py > 0.0 else "S" if py < 0.0 else ""
            title = f"{np.abs(py):.5f}° {pys} {np.abs(px):.5f}° {pxs}"
    else:
        _, attrs = geometry_containing_point(country_key, (x, y), mode)
        if attrs is not None:
            title = attrs["label"]
    return [html.H3(title)]


@APP.callback(
    Output("table_container", "children"),
    Input("issue_month", "value"),
    Input("freq", "value"),
    Input("mode", "value"),
    Input("geom_key", "data"),
    Input("location", "pathname"),
    Input("severity", "value"),
    Input("predictand", "value"),
    Input("predictors", "value"),
    Input("include_upcoming", "value"),
    State("season", "value"),
)
def table_cb(issue_month_abbrev, freq, mode, geom_key, pathname, severity, predictand_key, predictor_keys, include_upcoming, season_id):
    country_key = country(pathname)
    config = CONFIG["countries"][country_key]
    season_config = config["seasons"][season_id]
    issue_month0 = abbrev_to_month0[issue_month_abbrev]

    final_season = None
    if not include_upcoming:
        now = datetime.datetime.now()
        now = cftime.Datetime360Day(now.year, now.month, min(now.day, 30))
        final_season = (
            cftime.Datetime360Day(now.year, 1, 1) +
            pd.Timedelta(season_config["target_month"] * 30, unit='days')
        )
        # if the season's end is in the future
        if final_season + datetime.timedelta(days=season_config["length"] / 2 * 30) > now:
            final_season = cftime.Datetime360Day(
                final_season.year - 1, final_season.month, final_season.day
            )

    tcs = table_columns(
        config["datasets"],
        predictor_keys,
        predictand_key,
        season_config["length"],
    )

    try:
        if geom_key is None:
            raise NotFoundError("No region found")

        main_df, summary_df, thresholds = generate_tables(
            country_key,
            season_config,
            tcs,
            predictand_key,
            issue_month0,
            freq,
            mode,
            geom_key,
            final_season,
        )
        summary_presentation_df = format_summary_table(
            summary_df, tcs, thresholds,
            country_key, mode, freq, season_id, issue_month0,
            geom_key, severity,
        )
        return fbftable.gen_table(
            tcs, summary_presentation_df, main_df, thresholds, severity, final_season
        )
    except Exception as e:
        if isinstance(e, NotFoundError):
            # If it's the user just asked for a forecast that doesn't
            # exist yet, no need to log it.
            pass
        else:
            traceback.print_exc()
        # Return values that will blank out the table, so there's
        # nothing left over from the previous location that could be
        # mistaken for data for the current location.
        return None


@APP.callback(
    Output("freq", "className"),
    Input("severity", "value"),
)
def update_severity_color(value):
    return f"severity{value}"


@APP.callback(
    Output("raster_layer", "url"),
    Output("forecast_warning", "is_open"),
    Output("raster_colorbar", "colorscale"),
    Input("year", "value"),
    Input("issue_month", "value"),
    Input("freq", "value"),
    Input("location", "pathname"),
    Input("map_column", "value"),
    State("season", "value"),
)
def tile_url_callback(target_year, issue_month_abbrev, freq, pathname, map_col_key, season_id):
    colorscale = None  # default value in case an exception is raised
    try:
        country_key = country(pathname)
        country_config = CONFIG["countries"][country_key]
        target_month0 = country_config["seasons"][season_id]["target_month"]
        ds_configs = country_config["datasets"]
        ds_config = ds_configs["forecasts"].get(map_col_key)
        if ds_config is None:
            map_is_forecast = False
            ds_config = ds_configs["observations"][map_col_key]
        else:
            map_is_forecast = True
        issue_month0 = abbrev_to_month0[issue_month_abbrev]
        colorscale = ds_config.colormap.to_dash_leaflet()

        if map_is_forecast:
            # Check if we have the requested data so that if we don't, we
            # can explain why the map is blank.
            ds_configs["forecasts"][map_col_key].select(issue_month0, target_month0, target_year, freq)
            tile_url = f"{TILE_PFX}/forecast/{map_col_key}/{{z}}/{{x}}/{{y}}/{country_key}/{season_id}/{target_year}/{issue_month0}/{freq}"
        else:
            # As for forecast select above
            ds_configs["observations"][map_col_key].select(target_month0, target_year)
            tile_url = f"{TILE_PFX}/obs/{map_col_key}/{{z}}/{{x}}/{{y}}/{country_key}/{season_id}/{target_year}"
        error = False

    except Exception as e:
        tile_url = ""
        error = True
        if isinstance(e, NotFoundError):
            # If user asked for a forecast that hasn't been issued yet, no
            # need to log it.
            pass
        else:
            traceback.print_exc()

    return tile_url, error, colorscale


@APP.callback(
    Output("vuln_layer", "url"),
    Input("year", "value"),
    Input("location", "pathname"),
    Input("mode", "value"),
)
def _(year, pathname, mode):
    country_key = country(pathname)
    return f"{TILE_PFX}/vuln/{{z}}/{{x}}/{{y}}/{country_key}/{mode}/{year}"


@APP.callback(
    Output("borders", "data"),
    Input("location", "pathname"),
    Input("mode", "value"),
)
def borders(pathname, mode):
    if mode == "pixel":
        shapes = []
    else:
        country_key = country(pathname)
        # TODO We don't actually need vuln data, just reusing an existing
        # query function as an expediency. Year is arbitrary. Optimize
        # later.
        shapes = (
            retrieve_vulnerability(country_key, mode, 2020)
            ["the_geom"]
            .apply(shapely.geometry.mapping)
        )
    return {"features": shapes}


APP.clientside_callback(
    """
    function (
        mode, season, predictors, predictand, year, issue_month,
        freq, severity, include_upcoming, position, show_modal
    ) {
        args = {
            "mode": mode,
            "season": season,
            "predictors": predictors.join(" "),
            "predictand": predictand,
            "year": year,
            "issue_month": issue_month,
            "freq": freq,
            "severity": severity,
            "include_upcoming": include_upcoming,
            "position": JSON.stringify(position),
            "show_modal": show_modal,
        };
        return "?" + new URLSearchParams(args).toString();
    }
    """,
    Output("location", "search"),
    Input("mode", "value"),
    Input("season", "value"),
    Input("predictors", "value"),
    Input("predictand", "value"),
    Input("year", "value"),
    Input("issue_month", "value"),
    Input("freq", "value"),
    Input("severity", "value"),
    Input("include_upcoming", "value"),
    Input("marker",  "position"),
    Input("modal", "is_open")
)

# Endpoints


@SERVER.route(
    f"{TILE_PFX}/forecast/<forecast_key>/<int:tz>/<int:tx>/<int:ty>/<country_key>/<season_id>/<int:target_year>/<int:issue_month0>/<int:freq>"
)
def forecast_tile(forecast_key, tz, tx, ty, country_key, season_id, target_year, issue_month0, freq):
    config = CONFIG["countries"][country_key]
    season_config = config["seasons"][season_id]
    target_month0 = season_config["target_month"]

    da = config["datasets"]["forecasts"][forecast_key].select(issue_month0, target_month0, target_year, freq)
    p = tuple(config["marker"])
    if config.get("clip", True):
        clipping = lambda: geometry_containing_point(country_key, p, "0")[0]
    else:
        clipping = None
    resp = pingrid.tile(da, tx, ty, tz, clipping)
    return resp


@SERVER.route(
    f"{TILE_PFX}/obs/<obs_key>/<int:tz>/<int:tx>/<int:ty>/<country_key>/<season_id>/<int:target_year>"
)
def obs_tile(obs_key, tz, tx, ty, country_key, season_id, target_year):
    config = CONFIG["countries"][country_key]
    target_month0 = config["seasons"][season_id]["target_month"]
    da = config["datasets"]["observations"][obs_key].select(target_month0, target_year)
    p = tuple(CONFIG["countries"][country_key]["marker"])
    clipping, _ = geometry_containing_point(country_key, p, "0")
    resp = pingrid.tile(da, tx, ty, tz, clipping)
    return resp


@SERVER.route(
    f"{TILE_PFX}/vuln/<int:tz>/<int:tx>/<int:ty>/<country_key>/<mode>/<int:year>"
)
def vuln_tiles(tz, tx, ty, country_key, mode, year):
    im = produce_bkg_tile(Color(0, 0, 0, 0))
    if mode != "pixel":
        df = retrieve_vulnerability(country_key, mode, year)
        cfg = CONFIG["countries"][country_key]["datasets"]["vuln"]
        scale_min, scale_max = cfg["range"]
        shapes = [
            (
                r["the_geom"],
                pingrid.impl.DrawAttrs(
                    Color(255, 0, 0, 255),
                    pingrid.impl.with_alpha(
                        CMAPS[cfg["colormap"]].to_rgba_array()[
                            min(
                                255,
                                int(
                                    (r["normalized"] - scale_min)
                                    * 255
                                    / (scale_max - scale_min)
                                ),
                            )
                        ],
                        255,
                    )
                    if r["normalized"] is not None and not np.isnan(r["normalized"])
                    else Color(0, 0, 0, 0),
                    1,
                    cv2.LINE_AA,
                ),
            )
            for _, r in df.iterrows()
        ]
        im = pingrid.impl.produce_shape_tile(im, shapes, tx, ty, tz, oper="intersection")
    return pingrid.image_resp(im)


def produce_bkg_tile(
    background_color: Color,
    tile_width: int = 256,
    tile_height: int = 256,
) -> np.ndarray:
    im = np.zeros((tile_height, tile_width, 4), np.uint8) + background_color
    return im


@SERVER.route(f"{ADMIN_PFX}/stats")
def stats():
    ps = dict(
        pid=os.getpid(),
        active_count=threading.active_count(),
        current_thread_name=threading.current_thread().name,
        ident=threading.get_ident(),
        main_thread_ident=threading.main_thread().ident,
        stack_size=threading.stack_size(),
        threads={
            x.ident: dict(name=x.name, is_alive=x.is_alive(), is_daemon=x.daemon)
            for x in threading.enumerate()
        },
    )

    rs = dict(
        version=about.version,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        process_stats=ps,
    )
    return yaml_resp(rs)


# Do not imitate this. Use JSON responses, not YAML.
def yaml_resp(data):
    s = yaml.dump(data, default_flow_style=False, width=120, allow_unicode=True)
    resp = flask.Response(response=s, mimetype="text/x-yaml")
    resp.headers["Cache-Control"] = "private, max-age=0, no-cache, no-store"
    return resp


@SERVER.route(f"{PFX}/trigger_check")
def trigger_check():
    var = parse_arg("variable")
    country_key = parse_arg("country_key")
    mode = parse_arg("mode")
    season = parse_arg("season")
    issue_month0 = parse_arg("issue_month", int)
    season_year = parse_arg("season_year", int)
    freq = parse_arg("freq", float)
    thresh = parse_arg("thresh", float)
    bounds = parse_arg("bounds", default=None)
    region = parse_arg("region", default=None)

    config = CONFIG["countries"][country_key]
    if var in config["datasets"]["forecasts"]:
        var_is_forecast = True
        lower_is_worse = False
    elif var in config["datasets"]["observations"]:
        var_is_forecast = False
        lower_is_worse = config["datasets"]["observations"][var].lower_is_worse
    else:
        raise InvalidRequestError(f"Unknown variable {var}")

    if mode == "pixel":
        if bounds is None:
            raise InvalidRequestError("If mode is pixel then bounds must be provided")
        if region is not None:
            raise InvalidRequestError("If mode is pixel then region must not be provided")
    else:
        if bounds is not None:
            raise InvalidRequestError("If mode is {mode} then bounds must not be provided")
        if region is None:
            raise InvalidRequestError("If mode is {mode} then region must be provided")

    target_month0 = config["seasons"][season]["target_month"]

    if mode == "pixel":
        geom_key = bounds
    else:
        geom_key = region
    shape = region_shape(mode, country_key, geom_key)

    if var_is_forecast:
        data = config["datasets"]["forecasts"][var].select(
            issue_month0, target_month0, season_year, freq
        )
    else:
        data = config["datasets"]["observations"][var].select(target_month0, season_year)
    if 'lon' in data.coords:
        data = pingrid.average_over(data, shape, all_touched=True)



    value = data.item()
    if lower_is_worse:
        triggered = bool(value <= thresh)
    else:
        triggered = bool(value >= thresh)
    response = {
        "value": value,
        "triggered": triggered,
    }

    return response


@SERVER.route(f"{PFX}/<country_key>/export")
def export_endpoint(country_key):
    mode = parse_arg("mode", int) # not supporting pixel mode for now
    season_id = parse_arg("season")
    issue_month0 = parse_arg("issue_month0", int)
    freq = parse_arg("freq", float)
    geom_key = parse_arg("region")
    predictor_key = parse_arg("predictor")
    predictand_key = parse_arg("predictand")
    include_upcoming = parse_arg("include_upcoming", bool, default=False)

    config = CONFIG["countries"][country_key]

    ds_config = config["datasets"]

    forecast_keys = set(ds_config["forecasts"].keys())
    obs_keys = set(ds_config["observations"].keys())
    all_keys = forecast_keys | obs_keys
    if predictor_key not in all_keys:
        raise InvalidRequestError(f"Unsupported value {predictor_key} for predictor_key. Valid values are: {' '.join(all_keys)}")

    if predictand_key not in all_keys:
        raise InvalidRequestError(f"Unsupported value {predictand_key} for predictand_key. Valid values are: {' '.join(all_keys)}")

    season_config = config["seasons"].get(season_id)
    if season_config is None:
        seasons = ' '.join(config["seasons"].keys())
        raise InvalidRequestError(f"Unknown season {season}. Valid values are: {seasons}")

    target_month0 = season_config["target_month"]

    cols = table_columns(
        config["datasets"],
        [predictor_key],
        predictand_key,
        season_length=season_config["length"],
    )

    final_season = None
    if not include_upcoming:
        now = datetime.datetime.now()
        now = cftime.Datetime360Day(now.year, now.month, min(now.day, 30))
        final_season = (
            cftime.Datetime360Day(now.year, 1, 1) +
            pd.Timedelta(season_config["target_month"] * 30, unit='days')
        )
        # if the season's end is in the future
        if final_season + datetime.timedelta(days=season_config["length"] / 2 * 30) > now:
            final_season = cftime.Datetime360Day(
                final_season.year - 1, final_season.month, final_season.day
            )

    main_df, summary_df, thresholds = generate_tables(
        country_key,
        season_config,
        cols,
        predictand_key,
        issue_month0,
        freq,
        mode,
        geom_key,
        final_season,
    )

    main_df['year'] = main_df['time'].apply(lambda x: x.year)

    (worthy_action, act_in_vain, fail_to_act, worthy_inaction, accuracy) = (
        summary_df[predictor_key]
    )

    response = flask.jsonify({
        'skill': {
            'worthy_action': worthy_action,
            'act_in_vain': act_in_vain,
            'fail_to_act': fail_to_act,
            'worthy_inaction': worthy_inaction,
            'accuracy': accuracy,
        },
        'history': main_df[[
            'year',
            predictand_key, f"worst_{predictand_key}",
            predictor_key, f"worst_{predictor_key}"
        ]].to_dict('records'),
        'threshold': float(thresholds[predictor_key]),
    })
    return response


@SERVER.route(f"{PFX}/regions")
def regions_endpoint():
    country_key = parse_arg("country")
    level = parse_arg("level", int)

    shapes_config = CONFIG["countries"][country_key]["shapes"][level]
    query = sql.Composed([
        sql.SQL("with a as ("),
        sql.SQL(shapes_config["sql"]),
        sql.SQL(") select key, label from a"),
    ])
    with psycopg2.connect(**CONFIG["db"]) as conn:
        df = pd.read_sql(query, conn)
    d = {'regions': df.to_dict(orient="records")}
    return flask.jsonify(d)


if __name__ == "__main__":
    if CONFIG["mode"] != "prod":
        import warnings
        warnings.simplefilter("error")
        debug = True
    else:
        debug = False

    APP.run_server(
        CONFIG["dev_server_interface"],
        CONFIG["dev_server_port"],
        debug=debug,
        extra_files=os.environ["CONFIG"].split(";"),
        processes=CONFIG["dev_processes"],
        threaded=False,
    )
