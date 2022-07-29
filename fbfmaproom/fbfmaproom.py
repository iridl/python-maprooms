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
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry import Polygon, Point
from shapely.geometry.multipoint import MultiPoint
from psycopg2 import sql
import math
import traceback
import enum
import itertools
import uuid
import warnings

import __about__ as about
import pyaconf
import pingrid
from pingrid import BGRA, ClientSideError, InvalidRequestError, NotFoundError, parse_arg
import fbflayout
import fbftable
import dash_bootstrap_components as dbc

from collections import OrderedDict


config_files = os.environ["CONFIG"].split(":")

CONFIG = {}
for fname in config_files:
    CONFIG = pyaconf.merge([CONFIG, pyaconf.load(fname)])

DBPOOL = pingrid.init_dbpool("dbpool", CONFIG)

ZERO_SHAPE = [[[[0, 0], [0, 0], [0, 0], [0, 0]]]]

PFX = CONFIG["core_path"]
TILE_PFX = CONFIG["tile_path"]
ADMIN_PFX = CONFIG["admin_path"]

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


APP = FbfDash(
    __name__,
    external_stylesheets=[dbc.themes.SIMPLEX],
    server=SERVER,
    url_base_pathname=f"{PFX}/",
    meta_tags=[
        {"name": "description", "content": "content description 1234"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)
APP.title = "FBF--Maproom"

APP.layout = fbflayout.app_layout()


def table_columns(dataset_config, predictor_keys, predictand_key,
                  severity, season_length):
    format_funcs = {
        'year': lambda midpoint: year_label(midpoint, season_length),
        'number0': number_formatter(0),
        'number1': number_formatter(1),
        'number2': number_formatter(2),
        'number3': number_formatter(3),
        'number4': number_formatter(4),
        'timedelta_days': format_timedelta_days,
        'bad': format_bad,
        'enso': format_enso,
    }

    tcs = OrderedDict()
    tcs["time"] = dict(
        name="Year",
        format=format_funcs['year'],
        tooltip="The year whose forecast is displayed on the map",
        type=ColType.SPECIAL,
    )

    def make_column(key):
        if key in dataset_config['forecasts']:
            col_type = ColType.FORECAST
            ds_config = dataset_config['forecasts'][key]
        elif key in dataset_config['observations']:
            col_type = ColType.OBS
            ds_config = dataset_config['observations'][key]
        else:
            assert False, f'Unknown dataset key {key}'

        format_func = format_funcs[ds_config.get('format', 'number1')]
        if 'units' in ds_config:
            units = ds_config['units']
        elif col_type is ColType.OBS:
            units = open_obs_from_config(ds_config).attrs.get('units')
        elif col_type is ColType.FORECAST:
            units = open_forecast_from_config(ds_config).attrs.get('units')
        else:
            units = None
        return dict(
            name=ds_config['label'],
            units=units,
            format=format_func,
            tooltip=ds_config.get('description'),
            lower_is_worse=ds_config['lower_is_worse'],
            type=col_type,
        )

    for key in predictor_keys:
        tcs[key] = make_column(key)
    tcs[predictand_key] = make_column(predictand_key)

    return tcs


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


def nino_class(col_name, row, severity):
    if row[col_name] == 3:
        return f'cell-severity-{severity}'
    return ""


def data_path(relpath):
    return Path(CONFIG["data_root"], relpath)


def open_data_array(
    cfg,
    var_key,
    val_min=None,
    val_max=None,
):
    if var_key is None:
        da = xr.DataArray()
    else:
        try:
            da = (
                xr.open_zarr(data_path(cfg["path"]), consolidated=False)
                .rename({v: k for k, v in cfg["var_names"].items() if v})
                [var_key]
            )
        except Exception as e:
            raise Exception(f"Couldn't open {data_path(cfg['path'])}") from e

    # TODO: some datasets we pulled from ingrid already have colormap,
    # scale_max, and scale_min attributes. Should we just use those,
    # instead of getting them from the config file and/or computing
    # them?
    if val_min is None:
        if "range" in cfg:
            val_min = cfg["range"][0]
        else:
            assert False, "configuration doesn't specify range"
    if val_max is None:
        if "range" in cfg:
            val_max = cfg["range"][1]
        else:
            assert False, "configuration doesn't specify range"
    da.attrs["colormap"] = cfg["colormap"]
    da.attrs["scale_min"] = val_min
    da.attrs["scale_max"] = val_max
    return da


def open_vuln(country_key):
    dataset_key = "vuln"
    cfg = CONFIG["countries"][country_key]["datasets"][dataset_key]
    return open_data_array(
        cfg,
        None,
        val_min=None,
        val_max=None,
    )


def open_forecast(country_key, forecast_key):
    cfg = CONFIG["countries"][country_key]["datasets"]["forecasts"][forecast_key]
    return open_forecast_from_config(cfg)


def open_forecast_from_config(ds_config):
    return open_data_array(ds_config, "pne", val_min=0.0, val_max=100.0)


def open_obs(country_key, obs_key):
    cfg = CONFIG["countries"][country_key]["datasets"]["observations"][obs_key]
    return open_obs_from_config(cfg)


def open_obs_from_config(ds_config):
    return open_data_array(ds_config, "obs", val_min=0.0, val_max=1000.0)


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


def retrieve_geometry(
    country_key: str, point: Tuple[float, float], mode: str, year: Optional[int]
) -> Tuple[MultiPolygon, Dict[str, Any]]:
    df = retrieve_vulnerability(country_key, mode, year)
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
    dbpool = DBPOOL
    with dbpool.take() as cm:
        conn = cm.resource
        with conn:  # transaction
            s = sql.Composed(
                [
                    sql.SQL("with v as ("),
                    sql.SQL(sc["vuln_sql"]),
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
    df["the_geom"] = df["the_geom"].apply(
        lambda x: x if isinstance(x, MultiPolygon) else MultiPolygon([x])
    )
    return df


def generate_tables(
    country_key,
    season_id,
    table_columns,
    predictand_key,
    issue_month0,
    freq,
    mode,
    geom_key,
    severity,
):
    basic_ds = fundamental_table_data(country_key, table_columns,
                                      season_id, issue_month0,
                                      freq, mode, geom_key)
    if "pct" in basic_ds.coords:
        basic_ds = basic_ds.drop_vars("pct")
    basic_df = basic_ds.to_dataframe()
    main_df, summary_df, thresholds = augment_table_data(
        basic_df, freq, table_columns, predictand_key
    )
    if mode == 'pixel':
        bounds = geom_key
        region = ''
    else:
        bounds = ''
        region = geom_key
    summary_presentation_df = format_summary_table(
        summary_df, table_columns, thresholds,
        country_key, mode, freq, season_id, issue_month0,
        bounds, region, severity,
    )
    return main_df, summary_presentation_df


def get_mpoly(mode, country_key, geom_key):
    if mode == "pixel":
        [[y0, x0], [y1, x1]] = json.loads(geom_key)
        mpolygon = MultiPolygon([Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])])
    else:
        _, mpolygon = retrieve_geometry2(country_key, int(mode), geom_key)
    return mpolygon


def select_forecast(country_key, forecast_key, issue_month0, target_month0,
                    target_year=None, freq=None, mpolygon=None):
    l = (target_month0 - issue_month0) % 12

    da = open_forecast(country_key, forecast_key)

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
        da = da.sel(pct=freq)

    if mpolygon is not None:
        da = pingrid.average_over_trimmed(da, mpolygon, all_touched=True)
    return da



def select_obs(country_key, obs_keys, target_month0, target_year=None, mpolygon=None):
    ds = xr.Dataset(
        data_vars={
            obs_key: open_obs(country_key, obs_key)
            for obs_key in obs_keys
        }
    )
    if target_year is not None:
        target_date = (
            cftime.Datetime360Day(target_year, 1, 1) +
            pd.Timedelta(target_month0 * 30, unit='days')
        )
        try:
            ds = ds.sel(time=target_date)
        except KeyError:
            raise NotFoundError(f'No value for {" ".join(obs_keys)} on {target_date}') from None

    with warnings.catch_warnings():
        # ds.where in xarray 2022.3.0 uses deprecated numpy
        # functionality. A recent change deletes the offending line;
        # see if this catch_warnings can be removed once that's
        # released.
        # https://github.com/pydata/xarray/commit/3a320724100ab05531d8d18ca8cb279a8e4f5c7f
        warnings.filterwarnings("ignore", category=DeprecationWarning, module='numpy.core.fromnumeric')
        ds = ds.where(lambda x: x["time"].dt.month == target_month0 + 0.5, drop=True)
    if mpolygon is not None and 'lon' in ds.coords:
        ds = pingrid.average_over_trimmed(ds, mpolygon, all_touched=True)
    return ds


def fundamental_table_data(country_key, table_columns,
                           season_id, issue_month0, freq, mode,
                           geom_key):
    season_config = CONFIG["countries"][country_key]["seasons"][season_id]
    year_min, year_max = season_config["year_range"]
    season_length = season_config["length"]
    target_month0 = season_config["target_month"]
    mpolygon = get_mpoly(mode, country_key, geom_key)

    forecast_ds = xr.Dataset(
        data_vars={
            forecast_key: select_forecast(
                country_key, forecast_key, issue_month0, target_month0,
                freq=freq, mpolygon=mpolygon
            ).rename({'target_date':"time"})
            for forecast_key, col in table_columns.items()
            if col["type"] is ColType.FORECAST
        }
    )

    obs_keys = [key for key, col in table_columns.items() if col["type"] is ColType.OBS]
    obs_ds = select_obs(country_key, obs_keys, target_month0, mpolygon=mpolygon)

    main_ds = xr.merge(
        [
            forecast_ds,
            obs_ds,
        ]
    )

    year = main_ds["time"].dt.year
    main_ds = main_ds.where((year >= year_min) & (year <= year_max), drop=True)

    main_ds = main_ds.sortby("time", ascending=False)

    return main_ds


def augment_table_data(main_df, freq, table_columns, predictand_key):
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

    rank_pct = {
        key: regular_data[key].rank(method="min", ascending=is_ascending(key), pct=True)
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
        main_df[key] = regular_data[key]
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
        bounds,
        region,
        severity,
):
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
        'region': region,
        'severity': severity,
    }
    url = CONFIG["gantt_url"] + urllib.parse.urlencode(dict(data=json.dumps(args)))
    more_less = "less" if lower_is_worse else "greater"
    component = html.A(
        [
            dbc.Button(
                "Set trigger", color="info", id=id_, size='sm'
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
                         country, mode, freq, season_id, issue_month0, bounds, region, severity
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
                bounds,
                region,
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


@APP.callback(
    Output("logo", "src"),
    Output("map", "center"),
    Output("map", "zoom"),
    Output("marker", "position"),
    Output("season", "options"),
    Output("season", "value"),
    Output("vuln_colorbar", "colorscale"),
    Output("mode", "options"),
    Output("mode", "value"),
    Output("predictand", "options"),
    Output("predictand", "value"),
    Output("predictors", "options"),
    Output("predictors", "value"),
    Input("location", "pathname"),
)
def _(pathname):
    country_key = country(pathname)
    c = CONFIG["countries"][country_key]
    season_options = [
        dict(
            label=c["seasons"][k]["label"],
            value=k,
        )
        for k in sorted(c["seasons"].keys())
    ]
    season_value = min(c["seasons"].keys())
    x, y = c["marker"]
    cx, cy = c["center"]
    vuln_cs = pingrid.to_dash_colorscale(open_vuln(country_key).attrs["colormap"])
    mode_options = [
        dict(
            label=k["name"],
            value=str(i),
        )
        for i, k in enumerate(c["shapes"])
    ] + [dict(label="Pixel", value="pixel")]
    mode_value = "0"

    datasets_config = c["datasets"]
    predictors_options = predictand_options = [
        dict(
            label=v["label"],
            value=k,
        )
        for k, v in itertools.chain(
            datasets_config["forecasts"].items(),
            datasets_config["observations"].items()
        )
    ]
    predictors_value = datasets_config["defaults"]["predictors"]
    predictand_value = datasets_config["defaults"]["predictand"]

    return (
        f"{PFX}/custom/{c['logo']}",
        [cy, cx],
        c["zoom"],
        [y, x],
        season_options,
        season_value,
        vuln_cs,
        mode_options,
        mode_value,
        predictand_options,
        predictand_value,
        predictors_options,
        predictors_value,
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
    Input("location", "pathname"),
)
def _(season, pathname):
    country_key = country(pathname)
    c = CONFIG["countries"][country_key]["seasons"][season]
    year_min, year_max = c["year_range"]
    year_range = range(year_max, year_min - 1, -1)
    midpoints = [
        cftime.Datetime360Day(year, 1, 1) + pd.Timedelta(days=c["target_month"] * 30)
        for year in year_range
    ]
    year_options = [
        dict(
            label=year_label(midpoint, c["length"]),
            value=midpoint.year
        )
        for midpoint in midpoints
    ]
    year_value = year_max
    issue_month_options = [
        dict(
            label=pd.to_datetime(int(v) + 1, format="%m").month_name(),
            value=v,
        )
        for v in reversed(c["issue_months"])
    ]
    issue_month_value = c["issue_months"][-1]
    return (
        year_options,
        year_value,
        issue_month_options,
        issue_month_value,
    )


@APP.callback(
    Output("feature", "positions"),
    Output("geom_key", "data"),
    Input("location", "pathname"),
    Input("marker", "position"),
    Input("mode", "value"),
)
def update_selected_region(pathname, position, mode):
    country_key = country(pathname)
    y, x = position
    c = CONFIG["countries"][country_key]
    positions = None
    key = None
    if mode == "pixel":
        (x0, y0), (x1, y1) = calculate_bounds(
            (x, y), c["resolution"], c.get("origin", (0, 0))
        )
        pixel = MultiPoint([(x0, y0), (x1, y1)]).envelope
        geom, _ = retrieve_geometry(country_key, tuple(c["marker"]), "0", None)
        if pixel.intersects(geom):
            positions = [[[[y0, x0], [y1, x0], [y1, x1], [y0, x1]]]]
        key = str([[y0, x0], [y1, x1]])
    else:
        geom, attrs = retrieve_geometry(country_key, (x, y), mode, None)
        if geom is not None:
            positions = pingrid.mpoly_shapely_to_leaflet(geom)
            key = str(attrs["key"])
    if positions is None:
        positions = ZERO_SHAPE

    return positions, key




@APP.callback(
    Output("marker_popup", "children"),
    Input("location", "pathname"),
    Input("marker", "position"),
    Input("mode", "value"),
    Input("year", "value"),
)
def update_popup(pathname, position, mode, year):
    country_key = country(pathname)
    y, x = position
    c = CONFIG["countries"][country_key]
    title = "No Data"
    content = []
    if mode == "pixel":
        (x0, y0), (x1, y1) = calculate_bounds(
            (x, y), c["resolution"], c.get("origin", (0, 0))
        )
        pixel = MultiPoint([(x0, y0), (x1, y1)]).envelope
        geom, _ = retrieve_geometry(country_key, tuple(c["marker"]), "0", None)
        if pixel.intersects(geom):
            px = (x0 + x1) / 2
            pxs = "E" if px > 0.0 else "W" if px < 0.0 else ""
            py = (y0 + y1) / 2
            pys = "N" if py > 0.0 else "S" if py < 0.0 else ""
            title = f"{np.abs(py):.5f}° {pys} {np.abs(px):.5f}° {pxs}"
    else:
        _, attrs = retrieve_geometry(country_key, (x, y), mode, year)
        if attrs is not None:
            title = attrs["label"]
            fmt = lambda k: [html.B(k + ": "), attrs[k], html.Br()]
            content = (
                fmt("Vulnerability") + fmt("Mean") + fmt("Stddev") + fmt("Normalized")
            )
    return [html.H3(title), html.Div(content)]


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
    State("season", "value"),
)
def table_cb(issue_month0, freq, mode, geom_key, pathname, severity, predictand_key, predictor_keys, season_id):
    country_key = country(pathname)
    config = CONFIG["countries"][country_key]
    tcs = table_columns(
        config["datasets"],
        predictor_keys,
        predictand_key,
        severity,
        config["seasons"][season_id]["length"],
    )
    try:
        dft, dfs = generate_tables(
            country_key,
            season_id,
            tcs,
            predictand_key,
            issue_month0,
            freq,
            mode,
            geom_key,
            severity,
        )
        return fbftable.gen_table(tcs, dfs, dft, severity)
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
        return None, None


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
def tile_url_callback(target_year, issue_month0, freq, pathname, map_col_key, season_id):
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
        colorscale = pingrid.to_dash_colorscale(ds_config["colormap"])

        if map_is_forecast:
            # Check if we have the requested data so that if we don't, we
            # can explain why the map is blank.
            select_forecast(country_key, map_col_key, issue_month0, target_month0, target_year, freq)
            tile_url = f"{TILE_PFX}/forecast/{map_col_key}/{{z}}/{{x}}/{{y}}/{country_key}/{season_id}/{target_year}/{issue_month0}/{freq}"
        else:
            # As for select_forecast above
            select_obs(country_key, [map_col_key], target_month0, target_year)
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


# Endpoints


@SERVER.route(
    f"{TILE_PFX}/forecast/<forecast_key>/<int:tz>/<int:tx>/<int:ty>/<country_key>/<season_id>/<int:target_year>/<int:issue_month0>/<int:freq>"
)
def forecast_tile(forecast_key, tz, tx, ty, country_key, season_id, target_year, issue_month0, freq):
    season_config = CONFIG["countries"][country_key]["seasons"][season_id]
    target_month0 = season_config["target_month"]

    da = select_forecast(country_key, forecast_key, issue_month0, target_month0, target_year, freq)
    p = tuple(CONFIG["countries"][country_key]["marker"])
    clipping, _ = retrieve_geometry(country_key, p, "0", None)
    resp = pingrid.tile(da, tx, ty, tz, clipping)
    return resp


@SERVER.route(
    f"{TILE_PFX}/obs/<obs_key>/<int:tz>/<int:tx>/<int:ty>/<country_key>/<season_id>/<int:target_year>"
)
def obs_tile(obs_key, tz, tx, ty, country_key, season_id, target_year):
    season_config = CONFIG["countries"][country_key]["seasons"][season_id]
    target_month0 = season_config["target_month"]
    da = select_obs(country_key, [obs_key], target_month0, target_year)[obs_key]
    p = tuple(CONFIG["countries"][country_key]["marker"])
    clipping, _ = retrieve_geometry(country_key, p, "0", None)
    resp = pingrid.tile(da, tx, ty, tz, clipping)
    return resp


@SERVER.route(
    f"{TILE_PFX}/vuln/<int:tz>/<int:tx>/<int:ty>/<country_key>/<mode>/<int:year>"
)
def vuln_tiles(tz, tx, ty, country_key, mode, year):
    im = pingrid.produce_bkg_tile(BGRA(0, 0, 0, 0))
    da = open_vuln(country_key)
    if mode != "pixel":
        df = retrieve_vulnerability(country_key, mode, year)
        shapes = [
            (
                r["the_geom"],
                pingrid.DrawAttrs(
                    BGRA(0, 0, 255, 255),
                    pingrid.with_alpha(
                        pingrid.parse_colormap(da.attrs["colormap"])[
                            min(
                                255,
                                int(
                                    (r["normalized"] - da.attrs["scale_min"])
                                    * 255
                                    / (da.attrs["scale_max"] - da.attrs["scale_min"])
                                ),
                            )
                        ],
                        255,
                    )
                    if r["normalized"] is not None and not np.isnan(r["normalized"])
                    else BGRA(0, 0, 0, 0),
                    1,
                    cv2.LINE_AA,
                ),
            )
            for _, r in df.iterrows()
        ]
        im = pingrid.produce_shape_tile(im, shapes, tx, ty, tz, oper="intersection")
    return pingrid.image_resp(im)


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
    return pingrid.yaml_resp(rs)


@SERVER.route(f"{PFX}/pnep_percentile")
def pnep_percentile():
    """Let P(y) be the forecast probability of not exceeding the /freq/ percentile in year y.
    Let r be the rank of P(season_year) among all the P(y).
    Returns r divided by the number of forecast years times 100,
    unless the forecast for season_year is not yet available in which case it returns null."""
    # TODO better explanation

    country_key = parse_arg("country_key")
    mode = parse_arg("mode")
    season = parse_arg("season")
    issue_month0 = parse_arg("issue_month", int)
    season_year = parse_arg("season_year", int)
    freq = parse_arg("freq", float)
    prob_thresh = parse_arg("prob_thresh", float)
    bounds = parse_arg("bounds", required=False)
    region = parse_arg("region", required=False)

    forecast_key = "pnep"

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

    config = CONFIG["countries"][country_key]
    season_config = config["seasons"][season]

    target_month0 = season_config["target_month"]

    if mode == "pixel":
        geom_key = bounds
    else:
        geom_key = region
    mpoly = get_mpoly(mode, country_key, geom_key)

    try:
        pnep = select_forecast(country_key, forecast_key,issue_month0,
                               target_month0, season_year, freq,
                               mpolygon=mpoly)
    except KeyError:
        pnep = None

    if pnep is None:
        response = {
            "found": False,
        }
    else:
        forecast_prob = pnep.item()
        response = {
            "found": True,
            "probability": forecast_prob,
            "triggered": bool(forecast_prob >= prob_thresh),
        }

    return response


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
    bounds = parse_arg("bounds", required=False)
    region = parse_arg("region", required=False)

    config = CONFIG["countries"][country_key]
    if var in config["datasets"]["forecasts"]:
        var_is_forecast = True
        lower_is_worse = config["datasets"]["forecasts"][var]["lower_is_worse"]
    elif var in config["datasets"]["observations"]:
        var_is_forecast = False
        lower_is_worse = config["datasets"]["observations"][var]["lower_is_worse"]
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
    mpoly = get_mpoly(mode, country_key, geom_key)

    if var_is_forecast:
        data = select_forecast(country_key, var, issue_month0,
                               target_month0, season_year, freq,
                               mpolygon=mpoly)
    else:
        data = select_obs(country_key, [var], target_month0, season_year,
                          mpolygon=mpoly)[var]

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


def retrieve_geometry2(country_key: str, mode: int, region_key: str):
    config = CONFIG["countries"][country_key]
    sc = config["shapes"][mode]
    query = sql.Composed(
        [
            sql.SQL(
                "with a as (",
            ),
            sql.SQL(sc["sql"]),
            sql.SQL(") select the_geom, label from a where key::text = %(key)s"),
        ]
    )
    with DBPOOL.take() as cm:
        conn = cm.resource
        with conn:  # transaction
            df = pd.read_sql(query, conn, params={"key": region_key})
    if len(df) == 0:
        raise InvalidRequestError(f"invalid region {region_key}")
    assert len(df) == 1
    row = df.iloc[0]
    geom = wkb.loads(row["the_geom"].tobytes())
    return row["label"], geom


@SERVER.route(f"{PFX}/<country_key>/export")
def export_endpoint(country_key):
    mode = parse_arg("mode", int) # not supporting pixel mode for now
    season_id = parse_arg("season")
    issue_month0 = parse_arg("issue_month0", int)
    freq = parse_arg("freq", float)
    geom_key = parse_arg("region")
    predictor_key = parse_arg("predictor")
    predictand_key = parse_arg("predictand")

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

    mpoly = get_mpoly(mode, country_key, geom_key)

    cols = table_columns(
        config["datasets"],
        [predictor_key],
        predictand_key,
        severity=0, # unimportant because we won't be formatting it
        season_length=season_config["length"],
    )
    basic_ds = fundamental_table_data(
        country_key, cols, season_id, issue_month0,
        freq, mode, geom_key
    )
    if "pct" in basic_ds.coords:
        basic_ds = basic_ds.drop_vars("pct")
    basic_df = basic_ds.to_dataframe()
    main_df, summary_df, thresholds = augment_table_data(
        basic_df, freq, cols, predictand_key
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
    with DBPOOL.take() as cm:
        conn = cm.resource
        with conn:  # transaction
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
        extra_files=config_files,
        processes=CONFIG["dev_processes"],
        threaded=False,
    )
