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

import __about__ as about
import pyaconf
import pingrid
from pingrid import BGRA, ClientSideError, InvalidRequestError, NotFoundError, parse_arg
import fbflayout
import dash_bootstrap_components as dbc


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

APP = dash.Dash(
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


def table_columns(obs_config, obs_dataset_key):
    obs_dataset_names = {k: v["label"] for k, v in obs_config.items()}
    tcs = [
        dict(id="year_label", name="Year"),
        dict(id="enso_state", name="ENSO State"),
        dict(id="forecast", name="Forecast, %"),
        dict(id="obs_rank", name=f"{obs_dataset_names[obs_dataset_key]} Rank"),
        dict(id="bad_year", name="Reported Bad Years"),
    ]
    return tcs


def data_path(relpath):
    return Path(CONFIG["data_root"], relpath)


def open_data_array(
    cfg,
    country_key,
    var_key,
    val_min=None,
    val_max=None,
):
    if var_key is None:
        da = xr.DataArray()
    else:
        da = (
            xr.open_zarr(data_path(cfg["path"]), consolidated=False)
            .rename({v: k for k, v in cfg["var_names"].items() if v})
            [var_key]
        )
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
        country_key,
        None,
        val_min=None,
        val_max=None,
    )


def open_pnep(country_key):
    dataset_key = "pnep"
    cfg = CONFIG["countries"][country_key]["datasets"][dataset_key]
    return open_data_array(
        cfg,
        country_key,
        "pnep",
        val_min=0.0,
        val_max=100.0,
    )


def open_obs(country_key, obs_dataset_key):
    cfg = CONFIG["countries"][country_key]["datasets"]["observations"][obs_dataset_key]
    return open_data_array(
        cfg, country_key, "obs", val_min=0.0, val_max=1000.0
    )


ENSO_STATES = {
    1.0: "La Niña",
    2.0: "Neutral",
    3.0: "El Niño"
}


def fetch_enso():
    path = data_path(CONFIG["dataframes"]["enso"])
    ds = xr.open_zarr(path, consolidated=False)
    df = ds.to_dataframe()
    df["enso_state"] = df["dominant_class"].apply(lambda x: ENSO_STATES[x])
    df = df.drop("dominant_class", axis="columns")
    df = df.set_index(df.index.rename("time"))
    return df


def fetch_bad_years(country_key):
    config = CONFIG
    dbpool = DBPOOL
    dc = config["dataframes"]["bad_years"]
    with dbpool.take() as cm:
        conn = cm.resource
        with conn:  # transaction
            df = pd.read_sql_query(
                sql.SQL(
                    """
                    select month_since_01011960, bad_year2 as bad_year
                    from {schema}.{table}
                    where lower(adm0_name) = %(country_key)s
                    """
                ).format(
                    schema=sql.Identifier(dc["schema"]),
                    table=sql.Identifier(dc["table"]),
                ),
                conn,
                params={"country_key": country_key},
                # Using pandas' nullable boolean type. boolean is
                # different from bool, which is not nullable :-(
                dtype={"bad_year": "boolean"},
            )
    df["time"] = df["month_since_01011960"].apply(from_month_since_360Day)
    df = df.drop("month_since_01011960", axis=1)
    df = df.set_index("time")
    return df


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
    country_key: str, mode: str, year: Optional[int]
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
    obs_dataset_key,
    season_config,
    table_columns,
    issue_month0,
    freq,
    mode,
    geom_key,
    severity,
):
    basic_ds = fundamental_table_data(country_key, obs_dataset_key,
                                      season_config, issue_month0,
                                      freq, mode, geom_key)
    main_df, summary_df, thresholds = augment_table_data(basic_ds.to_dataframe(), freq)
    main_presentation_df = format_main_table(main_df, season_config["length"],
                                             table_columns, severity)
    summary_presentation_df = format_summary_table(summary_df, table_columns)
    # TODO no longer handling the case where geom_key is None. Does
    # that ever actually happen?
    return main_presentation_df, summary_presentation_df, thresholds

def merge_tables(summary, table):
    summary['summary'] = True
    table['summary'] = False
    return pd.concat([summary, table])


def get_mpoly(mode, country_key, geom_key):
    if mode == "pixel":
        [[y0, x0], [y1, x1]] = json.loads(geom_key)
        mpolygon = MultiPolygon([Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])])
    else:
        _, mpolygon = retrieve_geometry2(country_key, int(mode), geom_key)
    return mpolygon


def select_pnep(country_key, issue_month0, target_month0,
                target_year=None, freq=None, mpolygon=None):
    l = (target_month0 - issue_month0) % 12

    da = open_pnep(country_key)

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



def select_obs(country_key, obs_dataset_key, mpolygon=None):
    obs_da = open_obs(country_key, obs_dataset_key)
    obs_da = pingrid.average_over_trimmed(obs_da, mpolygon, all_touched=True)
    return obs_da


def fundamental_table_data(country_key, obs_dataset_key,
                           season_config, issue_month0, freq, mode,
                           geom_key):
    year_min, year_max = season_config["year_range"]
    season_length = season_config["length"]
    target_month = season_config["target_month"]
    mpolygon = get_mpoly(mode, country_key, geom_key)

    bad_years_df = fetch_bad_years(country_key)

    enso_df = fetch_enso()

    obs_da = select_obs(country_key, obs_dataset_key, mpolygon)
    obs_da = obs_da * season_length * 30 # TODO some datasets already aggregated over season?

    pnep_da = select_pnep(country_key, issue_month0, target_month,
                          freq=freq, mpolygon=mpolygon)
    main_ds = xr.Dataset(
        data_vars=dict(
            bad_year=bad_years_df["bad_year"].to_xarray(),
            obs=obs_da,
            pnep=pnep_da.rename({'target_date':"time"}),
        )
    )
    main_ds = xr.merge(
        [
            main_ds,
            enso_df["enso_state"].to_xarray()
        ],
        join="left"
    )

    year = main_ds["time"].dt.year
    main_ds = main_ds.where((year >= year_min) & (year <= year_max), drop=True)

    main_ds = main_ds.sortby("time", ascending=False)

    return main_ds


def augment_table_data(main_df, freq):
    main_df = pd.DataFrame(main_df)
    thresholds = {}

    obs = main_df["obs"].dropna()
    pnep = main_df["pnep"].dropna()
    bad_year = main_df["bad_year"].dropna().astype(bool)
    el_nino = main_df["enso_state"].dropna() == "El Niño"

    obs_rank = obs.rank(method="first", ascending=True)
    obs_rank_pct = obs.rank(method="first", ascending=True, pct=True)
    worst_obs = (obs_rank_pct <= freq / 100).astype(bool)
    thresholds["obs"] = obs_rank[worst_obs].max()

    pnep_max_rank_pct = pnep.rank(method="first", ascending=False, pct=True)
    worst_pnep = (pnep_max_rank_pct <= freq / 100).astype(bool)
    thresholds["pnep"] = pnep[worst_pnep].min()

    summary_df = pd.DataFrame.from_dict(dict(
        enso_state=hits_and_misses(el_nino, bad_year),
        forecast=hits_and_misses(worst_pnep, bad_year),
        obs_rank=hits_and_misses(worst_obs, bad_year),
    ))

    main_df["obs_rank"] = obs_rank

    return main_df, summary_df, thresholds


def format_pnep(x):
    if np.isnan(x):
        return ""
    return f"{x:.2f}"


def format_bad(x):
    # TODO some parts of the program use pandas boolean arrays, which
    # use pd.NA as the missing value indicator, while others use
    # xarray, which uses np.nan. This is annoying; I think we need to
    # stop using boolean arrays. They're marked experimental in the
    # pandas docs anyway.
    if pd.isna(x):
        return None
    if np.isnan(x):
        return None
    if x is True:
        return "Bad"
    if x is False:
        return ""
    raise Exception(f"Invalid bad_year value {x}")


def format_main_table(main_df, season_length, table_columns, severity):
    main_df = pd.DataFrame(main_df)
    midpoints = main_df.index.to_series()
    main_df["year_label"] = midpoints.apply(lambda x: year_label(x, season_length))

    main_df["forecast_numeric"] = main_df["pnep"]
    main_df["forecast"] = main_df["pnep"].apply(format_pnep)

    main_df["bad_year"] = main_df["bad_year"].apply(format_bad)

    # Discard unneeded columns, in the process also putting the
    # columns in the order the test expects.
    main_df = main_df[
        [c["id"] for c in table_columns] + ["forecast_numeric"]
    ]

    return main_df


def format_summary_table(summary_df, table_columns):
    summary_df = pd.DataFrame(summary_df)
    summary_df["year_label"] = [
        "Worthy-action:",
        "Act-in-vain:",
        "Fail-to-act:",
        "Worthy-Inaction:",
        "Rate:",
    ]
    headings_df = pd.DataFrame({c["id"]: [c["name"]] for c in table_columns})
    summary_df = summary_df.append(headings_df)
    return summary_df


def hits_and_misses(prediction, truth):
    assert pd.notnull(prediction).all()
    assert pd.notnull(truth).all()
    true_pos = (prediction & truth).sum()
    false_pos = (prediction & ~truth).sum()
    false_neg = (~prediction & truth).sum()
    true_neg = (~prediction & ~truth).sum()
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    return [true_pos, false_pos, false_neg, true_neg,
            f"{accuracy * 100:.2f}%"]


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
    Output("pnep_colorbar", "colorscale"),
    Output("vuln_colorbar", "colorscale"),
    Output("mode", "options"),
    Output("mode", "value"),
    Output("obs_datasets", "options"),
    Output("obs_datasets", "value"),
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
    pnep_cs = pingrid.to_dash_colorscale(open_pnep(country_key).attrs["colormap"])
    vuln_cs = pingrid.to_dash_colorscale(open_vuln(country_key).attrs["colormap"])
    mode_options = [
        dict(
            label=k["name"],
            value=str(i),
        )
        for i, k in enumerate(c["shapes"])
    ] + [dict(label="Pixel", value="pixel")]
    mode_value = "0"
    obs_datasets_cfg = c["datasets"]["observations"]
    obs_datasets_options = [
        dict(
            label=v["label"],
            value=k,
        )
        for k, v in obs_datasets_cfg.items()
    ]
    obs_datasets_value = next(iter(obs_datasets_cfg.keys()))
    return (
        f"{PFX}/custom/{c['logo']}",
        [cy, cx],
        c["zoom"],
        [y, x],
        season_options,
        season_value,
        pnep_cs,
        vuln_cs,
        mode_options,
        mode_value,
        obs_datasets_options,
        obs_datasets_value,
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
    Output("geom_key", "value"),
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

    # The leaflet Polygon class supports a MultiPolygon option, but
    # dash-leaflet's PropTypes rejects that format before Leaflet even
    # gets a chance to see it. Until we can get that fixed, using
    # the heuristic that the polygon with the most points in its outer
    # ring is the most important one.
    # https://github.com/thedirtyfew/dash-leaflet/issues/110
    heuristic_size = lambda polygon: len(polygon[0])
    longest_idx = max(
        range(len(positions)),
        key=lambda i: heuristic_size(positions[i])
    )
    positions = positions[longest_idx]

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
    Output("prob_thresh_text", "children"),
    Input("prob_thresh", "value"),
)
def display_prob_thresh(val):
    if val is not None:
        return f"{val:.2f}%"
    else:
        return ""


@APP.callback(
    Output("table", "children"),
    Output("prob_thresh", "value"),
    Input("issue_month", "value"),
    Input("freq", "value"),
    Input("mode", "value"),
    Input("geom_key", "value"),
    Input("location", "pathname"),
    Input("severity", "value"),
    Input("obs_datasets", "value"),
    State("season", "value"),
)
def _(issue_month0, freq, mode, geom_key, pathname, severity, obs_dataset_key, season):
    country_key = country(pathname)
    config = CONFIG["countries"][country_key]
    tcs = table_columns(config["datasets"]["observations"], obs_dataset_key)
    try:
        dft, dfs, thresholds = generate_tables(
            country_key,
            obs_dataset_key,
            config["seasons"][season],
            tcs,
            issue_month0,
            freq,
            mode,
            geom_key,
            severity,
        )
        rows = table_rows(dft, severity, thresholds)
        return rows, thresholds["pnep"]
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


def forecast_cell(val, severity, threshold):
    style = {}
    if val >= threshold:
        colorpair = fbflayout.SEVERITY_COLORS[severity]
        style["backgroundColor"] = colorpair.bg
        style["color"] = colorpair.fg
    return html.Td(f"{val:.2f}", style=style)


def obs_cell(val, severity, threshold):
    style = {}
    if val <= threshold:
        colorpair = fbflayout.SEVERITY_COLORS[severity]
        style["backgroundColor"] = colorpair.bg
        style["color"] = colorpair.fg
    if np.isnan(val):
        val_str = ""
    else:
        val_str = f"{val:.2f}"
    return html.Td(val_str, style=style)


def table_rows(table, severity, thresholds):
    rows = []
    for _, row in table.iterrows():
        rows.append(html.Tr([
            html.Td(row["year_label"]),
            forecast_cell(row["forecast_numeric"], severity, thresholds["pnep"]),
            obs_cell(row["obs_rank"], severity, thresholds["obs"]),
        ]))
    return rows


@APP.callback(
    Output("freq", "className"),
    Input("severity", "value"),
)
def update_severity_color(value):
    return f"severity{value}"


@APP.callback(
    Output("gantt", "href"),
    Input("issue_month", "value"),
    Input("freq", "value"),
    Input("geom_key", "value"),
    Input("mode", "value"),
    Input("year", "value"),
    Input("location", "pathname"),
    Input("severity", "value"),
    Input("prob_thresh", "value"),
    State("season", "value"),
)
def _(
    issue_month0,
    freq,
    geom_key,
    mode,
    year,
    pathname,
    severity,
    prob_thresh,
    season,
):
    country_key = country(pathname)
    config = CONFIG["countries"][country_key]
    season_config = config["seasons"][season]
    if mode == "pixel":
        region = None
        bounds = json.loads(geom_key)
    else:
        label = None
        try:
            label, _ = retrieve_geometry2(country_key, int(mode), geom_key)
        except:
            label = ""
        region = {
            "id": geom_key,
            "label": label,
        }
        bounds = None
    res = dict(
        country=country_key,
        mode=mode,
        season_year=year,
        freq=freq,
        prob_thresh=prob_thresh,
        season={
            "id": season,
            "label": season_config["label"],
            "target_month": season_config["target_month"],
            "length": season_config["length"],
        },
        issue_month=issue_month0,
        bounds=bounds,
        region=region,
        severity=severity,
    )
    # print("***:", res)
    url = CONFIG["gantt_url"] + urllib.parse.urlencode(dict(data=json.dumps(res)))
    return url


@APP.callback(
    Output("pnep_layer", "url"),
    Output("forecast_warning", "is_open"),
    Input("year", "value"),
    Input("issue_month", "value"),
    Input("freq", "value"),
    Input("location", "pathname"),
    State("season", "value"),
)
def pnep_tile_url_callback(target_year, issue_month0, freq, pathname, season_id):
    country_key = country(pathname)
    season_config = CONFIG["countries"][country_key]["seasons"][season_id]
    target_month0 = season_config["target_month"]

    try:
        # Prime the cache before the thundering horde of tile requests
        select_pnep(country_key, issue_month0, target_month0, target_year, freq)
        return f"{TILE_PFX}/pnep/{{z}}/{{x}}/{{y}}/{country_key}/{season_id}/{target_year}/{issue_month0}/{freq}", False
    except Exception as e:
        if isinstance(e, NotFoundError):
            # If user asked for a forecast that hasn't been issued yet, no
            # need to log it.
            pass
        else:
            traceback.print_exc()
        # if no raster data can be created then set the URL to be blank
        return "", True


@APP.callback(
    Output("vuln_layer", "url"),
    Input("year", "value"),
    Input("location", "pathname"),
    Input("mode", "value"),
)
def _(year, pathname, mode):
    country_key = country(pathname)
    if mode != "pixel":
        retrieve_vulnerability(country_key, mode, year)
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
    f"{TILE_PFX}/pnep/<int:tz>/<int:tx>/<int:ty>/<country_key>/<season_id>/<int:target_year>/<int:issue_month0>/<int:freq>"
)
def pnep_tiles(tz, tx, ty, country_key, season_id, target_year, issue_month0, freq):
    season_config = CONFIG["countries"][country_key]["seasons"][season_id]
    target_month0 = season_config["target_month"]

    da = select_pnep(country_key, issue_month0, target_month0, target_year, freq)
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
        pnep = select_pnep(country_key, issue_month0, target_month0, season_year,
                           freq, mpolygon=mpoly)
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


@SERVER.route(f"{PFX}/download_table")
def download_table():
    country_key = parse_arg("country_key")
    obs_dataset_key = parse_arg("obs_dataset_key")
    season_id = parse_arg("season_id")
    issue_month = parse_arg("issue_month")
    mode = parse_arg("mode")
    geom_key = parse_arg("geom_key")
    freq = parse_arg("freq", int, required=False)

    country_config = CONFIG["countries"][country_key]
    season_config = country_config["seasons"][season_id]
    issue_month0 = abbrev_to_month0[issue_month]

    tcs = table_columns(country_config["datasets"]["observations"], obs_dataset_key)

    main_ds = fundamental_table_data(
        country_key, obs_dataset_key, season_config, issue_month0, freq=freq,
        mode=mode, geom_key=geom_key
    )
    if freq is not None:
        augmented_df, _, thresholds = augment_table_data(main_ds.to_dataframe(), freq)
        main_ds["worst_pnep"] = (main_ds["pnep"] >= thresholds["pnep"]).astype(int)

    # flatten the 2d variable pnep into 19 1d variables pnep_05, pnep_10, ...
    if freq is None:
        freqs = range(5, 100, 5)
    else:
        freqs = [freq]
        # Add a pct dimension with a single value so that
        # .sel(pct=pct) below will work.
        main_ds["pnep"] = main_ds["pnep"].expand_dims("pct")
    for pct in freqs:
        main_ds[f'pnep_{pct:02}'] = main_ds["pnep"].sel(pct=pct)
    main_ds = main_ds.drop_vars(["pnep", "pct"])

    buf = io.StringIO()
    df = main_ds.to_dataframe()
    time = df.index.map(lambda x: x.strftime('%Y-%m-%d'))
    df["time"] = time
    df["bad_year"] = df["bad_year"].astype("float") # to acommodate NaN as missing value indicator

    cols = ["time", "bad_year", "obs", "enso_state"]
    if freq is not None:
        cols.append("worst_pnep")
    cols += [f"pnep_{pct:02}" for pct in freqs]
    df.to_csv(buf, columns=cols, index=False)
    output = flask.make_response(buf.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=export.csv"
    output.headers["Content-Type"] = "text/csv"
    return output


if __name__ == "__main__":
    if CONFIG["mode"] != "prod":
        import warnings
        warnings.simplefilter("error")
        debug = True
    else:
        debug = False

    APP.run_server(CONFIG["dev_server_interface"], CONFIG["dev_server_port"],
        debug=debug,
        extra_files=config_files,
    )
