__all__ = [
    'CORRELATION_CS',
    'CORRELATION_COLORMAP',
    'ClientSideError',
    'InvalidRequestError',
    'NotFoundError',
    'RAINBOW_COLORMAP',
    'RAINFALL_COLORMAP',
    'RAIN_PNE_COLORMAP',
    'RAIN_POE_COLORMAP',
    'average_over',
    'client_side_error',
    'deep_merge',
    'empty_tile',
    'error_fig',
    'image_resp',
    'load_config',
    'open_dataset',
    'open_mfdataset',
    'parse_arg',
    'parse_colormap',
    'sel_snap',
    'tile',
    'tile_left',
    'tile_top_mercator',
    'to_dash_colorscale',
    'to_dash_colorscale2',
]

import copy
import io
from typing import Tuple, List, Literal, Optional, Union, Callable, Iterable as Iterable
from typing import NamedTuple
import math
import datetime
import numpy as np
import pandas as pd
import xarray as xr
from collections.abc import Iterable as CollectionsIterable
import cv2
import psycopg2.extensions
from psycopg2 import sql
import rasterio.features
import rasterio.transform
import shapely.geometry
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.polygon import LinearRing
import urllib.parse
import flask
import yaml
import plotly.graph_objects as pgo
from datetime import datetime, timedelta
import werkzeug.datastructures

RAINBOW_COLORMAP = "[16711680 [16776960 51] [65280 51] [65535 51] [255 51] [16711935 51]]"
RAINFALL_COLORMAP = "[16777215 16777215 12632256 12632256 14155730 [14155730 21] 12841150 [12841150 23] 10215830 [10215830 23] 7262835 [7262835 22] 6931300 [6931300 23] 5944410 [5944410 23] 5285200 [5285200 23] 4625990 [4625990 22] 3966780 [3966780 23] 3307570 [3307570 23] 2648360 [2648360 23] 1989150]"
RAIN_POE_COLORMAP = "[0 [2763429 38] [42495 39] [65535 39] 11920639 [11920639 24] [3329330 3] [13688896 37] [16711680 38] [15736992 39]]"
RAIN_PNE_COLORMAP = "[15736992 [16711680 38] [13688896 39] [3329330 39] 11920639 [11920639 24] [65535 3] [42495 37] [2763429 38] [0 39]]"
CORRELATION_COLORMAP = "[8388608 [16711680 25] [16760576 26] [13959039 39] [10025880 26] 11920639 [11920639 26] 65535 [36095 38] [255 38] [128 39]]"


def sel_snap(spatial_array, lat, lng, dim_y="Y", dim_x="X"):
    """Selects the spatial_array's closest spatial grid center to the lng/lat coordinate.
    Raises an exception if lng/lat is outside spatial_array domain.
    Assumes regularly spaced dimensions.
    """
    the_method = None
    half_res_y = np.abs(spatial_array[dim_y][1] - spatial_array[dim_y][0]) / 2
    half_res_x = np.abs(spatial_array[dim_x][1] - spatial_array[dim_x][0]) / 2
    min_y = spatial_array[dim_y][[0, -1]].min()
    max_y = spatial_array[dim_y][[0, -1]].max()
    min_x = spatial_array[dim_x][[0, -1]].min()
    max_x = spatial_array[dim_x][[0, -1]].max()
    if lat >= min_y and lat <= max_y and lng >= min_x and lng <= max_x:
        the_method = "nearest"
    return spatial_array.sel(method=the_method, **{dim_x:lng, dim_y:lat})


def error_fig(error_msg="error"):
    return pgo.Figure().add_annotation(
        x=2,
        y=2,
        text=error_msg,
        font=dict(family="sans serif", size=30, color="crimson"),
        showarrow=False,
        yshift=10,
        xshift=60,
    )


FuncInterp2d = Callable[[Iterable[np.ndarray]], np.ndarray]


class ColorScale:
    
    def __init__(self, colors, scale=None):
        self.colors = colors
        self.scale = scale
    
    def cv2cm(self):
        if self.scale is None:
           cs_val = np.arange(len(colors))
        else:
           if len(self.colors) == len(self.scale):
               cs_val = np.array(self.scale)
           else:
               raise Exception("if provided, scale must be same length as colors")
        cv2cm_i = (255*(cs_val-cs_val[0])/(cs_val[-1]-cs_val[0])).astype(int)
        cv2cm_i = cv2cm_i + np.append([0], np.where(np.diff(cv2cm_i) == 0, 1, 0))
        cv2cm_rgba = np.array(self.colors)
        cv2cm_bgra = cv2cm_rgba[:,[2, 1, 0, 3]]
        cv2cm = np.full((256,4), np.nan)
        for bgra in range(4):
            cv2cm[:, bgra] = np.interp(np.arange(256), cv2cm_i, cv2cm_bgra[:, bgra])
        return cv2cm.astype(int)


class RGBA(NamedTuple):
    red: int
    green: int
    blue: int
    alpha: int = 255


class BGRA(NamedTuple):
    blue: int
    green: int
    red: int
    alpha: int


class DrawAttrs(NamedTuple):
    line_color: Union[int, BGRA]
    background_color: Union[int, BGRA]
    line_thickness: int
    line_type: int  # cv2.LINE_4 | cv2.LINE_8 | cv2.LINE_AA


def mercator_to_rad(lats: float) -> float:
    return np.arctan(0.5 * (np.exp(lats) - np.exp(-lats)))


def mercator_to_deg(lats: float) -> float:
    return np.rad2deg(mercator_to_rad(np.deg2rad(lats)))


def rad_to_mercator(lats: float) -> float:
    return np.log(np.tan(np.pi / 4 + lats / 2))


def deg_to_mercator(lats: float) -> float:
    return np.rad2deg(rad_to_mercator(np.deg2rad(lats)))


def nearest_interpolator(
    input_grids: Iterable[Tuple[float, float]],  # [(y0, dy), (x0, dx), ...]
    input_data: np.ndarray,
) -> FuncInterp2d:
    padded_data = np.pad(
        input_data, pad_width=1, mode="constant", constant_values=np.nan
    )

    def interp_func(output_grids: Iterable[np.ndarray]) -> np.ndarray:
        index = tuple(
            np.minimum(np.maximum(((x - (x0 - 1.5 * dx)) / dx).astype(int), 0), n - 1)
            for (x0, dx), x, n in zip(input_grids, output_grids, padded_data.shape)
        )
        return padded_data[tuple(reversed(np.meshgrid(*reversed(index))))]

    return interp_func


def create_interp(da: xr.DataArray) -> FuncInterp2d:
    x = da["lon"].values
    y = da["lat"].values
    input_grids = [
        (y[0], y[1] - y[0]),
        (x[0], x[1] - x[0]),
    ]  # require at least 2 points in each spatial dimension, and assuming that the grid is even
    input_data = da.transpose("lat", "lon").values
    f = nearest_interpolator(input_grids, input_data)
    return f


def from_months_since(x, year_since=1960):
    int_x = int(x)
    return datetime.date(
        int_x // 12 + year_since, int_x % 12 + 1, int(30 * (x - int_x) + 1)
    )


from_months_since_v = np.vectorize(from_months_since)


def to_months_since(d, year_since=1960):
    return (d.year - year_since) * 12 + (d.month - 1) + (d.day - 1) / 30.0


def tile_left(tx: int, tz: int) -> float:
    """"Maps a column number in the tile grid at scale z to the longitude
    of the left edge of that tile in degrees. Appropriate for both Mercator
    and equirectangular projections.
    """
    return tx * 360 / 2 ** tz - 180


# Commenting this out because it's currently not used and I'm not sure
# it's correct. In the equirectangular projection, the grid isn't
# square: it's 360 degrees across and only 180 degrees from top to
# bottom. We would need to use rectangular tiles instead of squares,
# which I think means either the above function or this one needs to
# be modified to change the aspect ratio.
# def g_lat(ty: int, tz: int) -> float:
#     """"Maps a row number in the equirectangular tile grid at scale z, to
#     the latitude of the bottom edge of that row in degrees."""
#     return ty * 180 / 2 ** tz - 90


def tile_top_mercator(ty: int, tz: int) -> float:
    """"Maps a row number in the spherical Mercator tile grid at scale z
    to the latitude of the top edge of that row in degrees.
    """
    a = math.pi - 2 * math.pi * ty / 2 ** tz
    return np.rad2deg(mercator_to_rad(a))


def pixel_extents(g: Callable[[int, int], float], tx: int, tz: int, n: int = 1):
    """Given a function that maps a tile coordinate (row or column number)
    to the start of that tile (top or left edge) in degrees, returns
    the bounds of each pixel within the tile along that dimension.

    """
    assert n >= 1 and tz >= 0 and 0 <= tx < 2 ** tz
    a = g(tx, tz)
    for i in range(1, n + 1):
        b = g(tx + i / n, tz)
        yield a, b
        a = b


def tile(da, tx, ty, tz, clipping=None):
    image_array = _tile(da, tx, ty, tz, clipping)
    return image_resp(image_array)


def _tile(da, tx, ty, tz, clipping):
    z = produce_data_tile(da, tx, ty, tz)
    if z is None:
        return empty_tile()

    if da.attrs["test"]:
        im = apply_colormap(
            z,
            da.attrs["colormap"].cv2cm(),
            da.attrs["scale_min"],
            da.attrs["scale_max"],
        ) 
    else:
        im = apply_colormap(
            z,
            parse_colormap(da.attrs["colormap"]),
            da.attrs["scale_min"],
            da.attrs["scale_max"],
        )
    if clipping is not None:
        if callable(clipping):
            clipping = clipping()
        draw_attrs = DrawAttrs(
            BGRA(0, 0, 255, 255), BGRA(0, 0, 0, 0), 1, cv2.LINE_AA
        )
        shapes = [(clipping, draw_attrs)]
        im = produce_shape_tile(im, shapes, tx, ty, tz, oper="difference")

    return im


def empty_tile(width: int = 256, height: int = 256):
    # If tile size were hard-coded, this could be a constant instead
    # of a function, but we're keeping open the option of changing
    # tile size. Also, numpy arrays are mutable, and having a mutable
    # global constant could lead to tricky bugs.
    im = apply_colormap(
        np.full([height, width], np.nan),
        np.zeros((256, 4)),
        # arbitrary min and max
        scale_min=0,
        scale_max=0,
    )
    return im


def produce_data_tile(
    da: xr.DataArray,
    tx: int,
    ty: int,
    tz: int,
    tile_width: int = 256,
    tile_height: int = 256,
) -> np.ndarray:
    x = np.fromiter(
        (a + (b - a) / 2.0 for a, b in pixel_extents(tile_left, tx, tz, tile_width)),
        np.double,
    )
    y = np.fromiter(
        (a + (b - a) / 2.0 for a, b in pixel_extents(tile_top_mercator, ty, tz, tile_height)),
        np.double,
    )
    tile_bbox = shapely.geometry.box(x[0], y[0], x[-1], y[-1])
    lon = da['lon']
    lat = da['lat']
    da_bbox = shapely.geometry.box(lon[0], lat[0], lon[-1], lat[-1])
    if tile_bbox.intersects(da_bbox):
        interp = create_interp(da)
        z = interp([y, x])
    else:
        z = None
    return z


def image_resp(im):
    cv2_imencode_success, buffer = cv2.imencode(".png", im)
    assert cv2_imencode_success
    io_buf = io.BytesIO(buffer)
    resp = flask.send_file(io_buf, mimetype="image/png")
    return resp


def to_multipolygon(p: Union[Polygon, MultiPolygon]) -> MultiPolygon:
    if not isinstance(p, MultiPolygon):
        p = MultiPolygon([p])
    return p


def rasterize_linearring(
    im: np.ndarray,
    ring: LinearRing,
    fxs: Callable[[np.ndarray], np.ndarray] = lambda xs: xs,
    fys: Callable[[np.ndarray], np.ndarray] = lambda ys: ys,
    line_type: int = cv2.LINE_AA,  # cv2.LINE_4 | cv2.LINE_8 | cv2.LINE_AA,
    color: Union[int, BGRA] = 255,
    shift: int = 0,
) -> np.ndarray:
    if not ring.is_empty:
        xs, ys = ring.coords.xy
        xs = fxs(np.array(xs)).astype(np.int32)
        ys = fys(np.array(ys)).astype(np.int32)
        pts = np.column_stack((xs, ys))
        pts = pts.reshape((1,) + pts.shape)
        cv2.fillPoly(im, pts, color, line_type, shift)
    return im


def rasterize_multipolygon(
    im: np.ndarray,
    mp: MultiPolygon,
    fxs: Callable[[np.ndarray], np.ndarray] = lambda xs: xs,
    fys: Callable[[np.ndarray], np.ndarray] = lambda ys: ys,
    line_type: int = cv2.LINE_AA,  # cv2.LINE_4 | cv2.LINE_8 | cv2.LINE_AA,
    fg_color: Union[int, BGRA] = 255,
    bg_color: Union[int, BGRA] = 0,
    shift: int = 0,
) -> np.ndarray:
    for p in mp.geoms:
        if not p.is_empty:
            rasterize_linearring(im, p.exterior, fxs, fys, line_type, fg_color, shift)
            for q in p.interiors:
                rasterize_linearring(im, q, fxs, fys, line_type, bg_color, shift)
    return im


def flatten(im_fg: np.ndarray, im_bg: np.ndarray) -> np.ndarray:
    # fg = mask
    # bg = unmasked part of the image
    # c = bgr
    # a = alpha = opacity
    im_fg = im_fg.astype(np.float64) / 255.0
    im_bg = im_bg.astype(np.float64) / 255.0
    c_fg = im_fg[:, :, :3]
    a_fg = im_fg[:, :, 3:]
    c_bg = im_bg[:, :, :3]
    a_bg = im_bg[:, :, 3:]
    a_comp = a_fg + (1.0 - a_fg) * a_bg
    # Avoid division by zero. If alpha is zero, it doesn't matter what
    # values b, g, r have; arbitrarily using 1.
    denom = np.where(a_comp > 0, a_comp, 1)
    c_comp = (a_fg * c_fg + (1.0 - a_fg) * a_bg * c_bg) / denom
    im_comp = np.concatenate((c_comp, a_comp), axis=2) * 255.0
    return im_comp.astype(np.uint8)


def apply_mask(
    im: np.ndarray, mask: np.ndarray, mask_color: BGRA = BGRA(0, 0, 0, 0)
) -> np.ndarray:
    h = im.shape[0]
    w = im.shape[1]
    mask = mask.reshape(mask.shape + (1,)).astype(np.float64) / 255
    mask_color = np.array(mask_color, np.float64).reshape((1, 1, 4))
    im_fg = mask_color * mask
    im_bg = im * (1.0 - mask)
    im_comp = flatten(im_fg, im_bg)
    return im_comp


def produce_shape_tile(
    im: np.ndarray,
    shapes: List[Tuple[MultiPolygon, DrawAttrs]],
    tx: int,
    ty: int,
    tz: int,
    oper: Literal["intersection", "difference"] = "intersection",
) -> np.ndarray:
    tile_height = im.shape[0]
    tile_width = im.shape[1]

    x0, x1 = list(pixel_extents(tile_left, tx, tz, 1))[0]
    y0, y1 = list(pixel_extents(tile_top_mercator, ty, tz, 1))[0]

    x_ratio = tile_width / (x1 - x0)
    y0_mercator = deg_to_mercator(y0)
    y_ratio_mercator = tile_height / (deg_to_mercator(y1) - y0_mercator)

    tile_bounds = (x0, y0, x1, y1)
    tile = MultiPoint([(x0, y0), (x1, y1)]).envelope

    for s, a in shapes:
        mask = np.zeros(im.shape[:2], np.uint8)
        if oper == "difference":
            if tile.intersects(s):
                mp = to_multipolygon(tile.difference(s))
            else:
                mp = to_multipolygon(tile)
        elif oper == "intersection":
            if tile.intersects(s):
                mp = to_multipolygon(tile.intersection(s))
            else:
                continue
        fxs = lambda xs: (xs - x0) * x_ratio
        fys = lambda ys: (deg_to_mercator(ys) - y0_mercator) * y_ratio_mercator
        rasterize_multipolygon(mask, mp, fxs, fys, a.line_type, 255, 0)
        im = apply_mask(im, mask, a.background_color)

    return im


AQUAMARINE = RGBA(127, 255, 212)
BLUE = RGBA(0, 0, 255)
DARKORANGE = RGBA(255, 140, 0)
DARKRED = RGBA(128, 0, 0)
DEEPSKYBLUE = RGBA(0, 191, 255)
MOCCASIN = RGBA(255, 228, 181)
NAVY = RGBA(0, 0, 128)
PALEGREEN = RGBA(152, 251, 152)
RED = RGBA(255, 0, 0)
YELLOW = RGBA(255, 255, 0)

CORRELATION_CS = ColorScale(
    [NAVY, BLUE, DEEPSKYBLUE, AQUAMARINE, PALEGREEN, MOCCASIN, MOCCASIN, YELLOW, DARKORANGE, RED, DARKRED],
    [-1, -0.8, -0.6, -0.3, -0.1, -0.1, 0.1, 0.1, 0.4, 0.7, 1],
)


def colorscale2cv2colormap(colorscale):
    """Returns a cv2 colormap from a pingrid `colorscale` .

    pingrid colorscale is a dictionary of RGBA colors as keys
    and numerical thresholds as values.
    Those thresholds must be and array of 1 or 2 elements.
    Two elements indicate a color band between the 2 values,
    while otherwise colors are blended from value to value.

    Parameters
    ----------
    colorscale : dict
        a pingrid colorscale dictionary

    Returns
    -------
    cv2cm : np.array[int]
        a BGRA 4x256 cv2 colormap array
    """
    cs_val = np.array([
        list(colorscale.values())[i][j]
        for i in range(len(colorscale))
        for j in range(len(list(colorscale.values())[i]))
    ])
    cv2cm_i = (255*(cs_val-cs_val[0])/(cs_val[-1]-cs_val[0])).astype(int)
    cv2cm_i = cv2cm_i + np.append([0], np.where(np.diff(cv2cm_i) == 0, 1, 0))
    cv2cm_rgba = np.array([
        list(colorscale.keys())[i]
        for i in range(len(colorscale))
        for j in range(len(list(colorscale.values())[i]))
    ])
    cv2cm_bgra = cv2cm_rgba[:,[2, 1, 0, 3]]
    cv2cm = np.full((256,4), np.nan)
    for colors in range(4):
        cv2cm[:,colors] = np.interp(np.arange(256), cv2cm_i, cv2cm_bgra[:,colors])
    return cv2cm.astype(int)


def parse_color(s: str) -> BGRA:
    v = int(s, 0)  # 0 tells int() to guess radix
    return BGRA(v >> 16 & 0xFF, v >> 8 & 0xFF, v >> 0 & 0xFF, 255)


def parse_color_item(vs: List[BGRA], s: str) -> List[BGRA]:
    if s == "null":
        rs = [BGRA(0, 0, 0, 0)]
    elif s[0] == "[":
        rs = [parse_color(s[1:])]
    elif s[-1] == "]":
        n = int(s[:-1])
        assert 1 < n <= 255 and len(vs) >= 2
        first = vs[-2]
        last = vs[-1]
        vs = vs[:-2]
        rs = [
            BGRA(
                first.blue + (last.blue - first.blue) * i / n,
                first.green + (last.green - first.green) * i / n,
                first.red + (last.red - first.red) * i / n,
                255
            )
            for i in range(n + 1)
        ]
    else:
        rs = [parse_color(s)]
    return vs + rs


def parse_colormap(s: str) -> np.ndarray:
    "Converts an Ingrid colormap to a cv2 colormap"
    vs = []
    for x in s[1:-1].split(" "):
        vs = parse_color_item(vs, x)
    # print(
    #     "*** CM cm:",
    #     len(vs),
    #     [f"{v.red:02x}{v.green:02x}{v.blue:02x}{v.alpha:02x}" for v in vs],
    # )
    rs = np.array([vs[int(i / 256.0 * len(vs))] for i in range(0, 256)], np.uint8)
    return rs


def to_dash_colorscale(s: str) -> List[str]:
    "Converts an Ingrid colormap to a dash colorscale"
    cm = parse_colormap(s)
    cs = []
    for x in cm:
        v = BGRA(*x)
        cs.append(f"#{v.red:02x}{v.green:02x}{v.blue:02x}{v.alpha:02x}")
    return cs


def to_dash_colorscale2(s):
    cm = s.cv2cm()
    cs = []
    for x in cm:
        v = BGRA(*x)
        cs.append(f"#{v.red:02x}{v.green:02x}{v.blue:02x}{v.alpha:02x}")
    return cs

def apply_colormap(x: np.ndarray, colormap: np.ndarray,
                   scale_min: float, scale_max: float) -> np.ndarray:
    im = (
        (x - scale_min) * 255 /
        (scale_max- scale_min)
    ).clip(0, 255)

    # int arrays have no missing value indicator, so record where the
    # NaNs were before casting to int.
    mask = np.isnan(im)
    im = im.astype(np.uint8)
    im = cv2.merge(
        [
            cv2.LUT(im, colormap[:, 0]),
            cv2.LUT(im, colormap[:, 1]),
            cv2.LUT(im, colormap[:, 2]),
            np.where(mask, 0, cv2.LUT(im, colormap[:, 3])),
        ]
    )
    return im


def with_alpha(c: BGRA, alpha) -> BGRA:
    return BGRA(*c[:3], alpha)


#
# Functions to deal with spatial averaging
#


def trim_to_bbox(ds, s, lon_name="lon", lat_name="lat"):
    """Given a Dataset and a shape, return the subset of the Dataset that
    intersects the shape's bounding box.
    """
    lon_res = ds[lon_name].values[1] - ds[lon_name].values[0]
    lat_res = ds[lat_name].values[1] - ds[lat_name].values[0]

    lon_min, lat_min, lon_max, lat_max = s.bounds
    # print("*** shape bounds:", lon_min, lat_min, lon_max, lat_max)

    lon_min -= 1 * lon_res
    lon_max += 1 * lon_res
    lat_min -= 1 * lat_res
    lat_max += 1 * lat_res

    return ds.sel(
        {lon_name: slice(lon_min, lon_max), lat_name: slice(lat_min, lat_max)}
    )


def average_over(ds, s, lon_name="lon", lat_name="lat", all_touched=False):
    """Average a Dataset over a shape"""
    lon_res = ds[lon_name].values[1] - ds[lon_name].values[0]
    lat_res = ds[lat_name].values[1] - ds[lat_name].values[0]

    ds = trim_to_bbox(ds, s, lon_name=lon_name, lat_name=lat_name)

    lon_min = ds[lon_name].values[0] - 0.5 * lon_res
    lon_max = ds[lon_name].values[-1] + 0.5 * lon_res
    lat_min = ds[lat_name].values[0] - 0.5 * lat_res
    lat_max = ds[lat_name].values[-1] + 0.5 * lat_res

    lon_size = ds.sizes[lon_name]
    lat_size = ds.sizes[lat_name]

    t = rasterio.transform.Affine(
        (lon_max - lon_min) / lon_size,
        0,
        lon_min,
        0,
        (lat_max - lat_min) / lat_size,
        lat_min,
    )

    r0 = rasterio.features.rasterize(
        [s], out_shape=(lat_size, lon_size), transform=t, all_touched=all_touched
    )
    r0 = xr.DataArray(
        r0,
        dims=(lat_name, lon_name),
        coords={lat_name: ds[lat_name], lon_name: ds[lon_name]},
    )
    r = r0 * np.cos(np.deg2rad(ds[lat_name]))

    res = ds.weighted(r).mean([lat_name, lon_name], skipna=True)

    # For some reason, DataArray names get preserved when they're
    # inside a Dataset, but not when ds itself is a DataArray.
    if isinstance(res, xr.DataArray):
        res.name = ds.name

    return res


#
# Functions to deal with periodic dimension (e.g. longitude)
#


def __dim_range(ds, dim, period=360.0):
    c0, c1 = ds[dim].values[0], ds[dim].values[-1]
    d = (period - (c1 - c0)) / 2.0
    c0, c1 = c0 - d, c1 + d
    return c0, c1


def __normalize_vals(v0, vals, period=360.0, right=False):

    vs = vals if isinstance(vals, CollectionsIterable) else [vals]

    v1 = v0 + period
    assert v0 <= 0.0 <= v1

    vs = np.mod(vs, period)
    if right:
        vs[vs > v1] -= period
    else:
        vs[vs >= v1] -= period

    vs = vs if isinstance(vals, CollectionsIterable) else vs[0]

    return vs


def __normalize_dim(ds, dim, period=360.0):
    """Doesn't copy ds. Make a copy if necessary."""
    c0, c1 = __dim_range(ds, dim, period)
    if c0 > 0.0:
        ds[dim] = ds[dim] - period
    elif c1 < 0.0:
        ds[dim] = ds[dim] + period


def roll_to(ds, dim, val, period=360.0):
    """Rolls the ds to the first dim's label that is greater or equal to
    val, and then makes dim monitonically increasing. Assumes that dim
    is monotonically increasing, covers exactly one period, and overlaps
    val. If val is outside of the dim, this function does nothing.
    """
    a = np.argwhere(ds[dim].values >= val)
    n = a[0, 0] if a.shape[0] != 0 else 0
    if n != 0:
        ds = ds.copy()
        ds = ds.roll(**{dim: -n}, roll_coords=True)
        ds[dim] = xr.where(ds[dim] < val, ds[dim] + period, ds[dim])
        __normalize_dim(ds, dim, period)
    return ds


def sel_periodic(ds, dim, vals, period=360.0):
    """Assumes that dim is monotonically increasing, covers exactly one period, and overlaps 0.0
    Examples: lon: 0..360, -180..180, -90..270, -360..0, etc.
    TODO: change API to match xarray's `sel`
    """
    c0, c1 = __dim_range(ds, dim, period)
    print(f"*** sel_periodic (input): {c0}..{c1}: {vals}")

    if isinstance(vals, slice):
        if vals.step is None or vals.step >= 0:
            s0 = __normalize_vals(c0, vals.start, period)
            s1 = __normalize_vals(c0, vals.stop, period, True)
        else:
            s0 = __normalize_vals(c0, vals.stop, period)
            s1 = __normalize_vals(c0, vals.start, period, True)

        print(f"*** sel_periodic (normalized): {c0}..{c1}: {s0=}, {s1=}")

        if s0 > s1:
            ds = roll_to(ds, dim, s1, period)
            c0, c1 = __dim_range(ds, dim, period)
            s0 = __normalize_vals(c0, s0, period)
            s1 = __normalize_vals(c0, s1, period, True)
            print(f"*** sel_periodic (rolled): {c0}..{c1}: {s0=}, {s1=}")

        if vals.step is None or vals.step >= 0:
            vals = slice(s0, s1, vals.step)
        else:
            vals = slice(s1, s0, vals.step)

        print(f"*** sel_periodic (slice): {c0}..{c1}: {vals}")

    else:
        vals = __normalize_vals(c0, vals, period=period)
        print(f"*** sel_periodic (array): {c0}..{c1}: {vals}")

    ds = ds.sel({dim: vals})

    return ds



# Flask utils


class ClientSideError(Exception):
    def __init__(self, message, status):
        self.message = message
        self.status = status
        super().__init__(message)

    def to_dict(self):
        return {
            "status": self.status,
            "name": type(self).__name__,
            "message": self.message,
        }


class InvalidRequestError(ClientSideError):
    def __init__(self, message):
        super().__init__(message, 400)


class NotFoundError(ClientSideError):
    def __init__(self, message):
        super().__init__(message, 404)


def client_side_error(e):
    return (e.to_dict(), e.status)


REQUIRED = object()

def parse_arg(name, conversion=str, default=REQUIRED, qstring=None):
    '''Stricter version of flask.request.args.get. Raises an exception in
cases where args.get ignores the problem and silently falls back on a
default behavior:

    - if type conversion fails
    - if the same arg is specified multiple times
    - if a required arg is not provided
    '''
    if qstring is None:
        args = flask.request.args
    else:
        if qstring == "":
            args = werkzeug.datastructures.ImmutableMultiDict({})
        else:
            assert qstring[0] == "?"
            args = urllib.parse.parse_qs(
                qstring[1:],
                strict_parsing=True
            )
            args = werkzeug.datastructures.ImmutableMultiDict(args)
    raw_vals = args.getlist(name)
    if len(raw_vals) > 1:
        raise InvalidRequestError(f"{name} was provided multiple times")
    if len(raw_vals) == 0:
        if default is REQUIRED:
            raise InvalidRequestError(f"{name} is required")
        else:
            return default
    try:
        val = conversion(raw_vals[0])
    except Exception as e:
        raise InvalidRequestError(f"{name} must be interpretable as {conversion}: {e}") from e

    return val


def fix_calendar(ds):
    for name, coord in ds.coords.items():
        if coord.attrs.get("calendar") == "360":
            coord.attrs["calendar"] = "360_day"
    ds = xr.decode_cf(ds)
    return ds


def open_dataset(*args, **kwargs):
    """Open a dataset with xarray, fixing incorrect CF metadata generated
    by Ingrid."""
    return _proxy(xr.open_dataset, *args, **kwargs)


def open_mfdataset(*args, **kwargs):
    """Open a multi-file dataset with xarray, fixing incorrect CF metadata generated
    by Ingrid."""
    return _proxy(xr.open_mfdataset, *args, **kwargs)


def _proxy(fn, *args, **kwargs):
    decode_cf = kwargs.get("decode_cf", True)
    decode_times = kwargs.pop("decode_times", decode_cf)
    if decode_times and not decode_cf:
        raise Exception("Don't know how to decode_times without decode_cf.")
    ds = fn(*args, decode_times=False, **kwargs)
    if decode_times:
        ds = fix_calendar(ds)
    return ds


# Copyright tfeldmann, MIT license.
# https://gist.github.com/angstwad/bf22d1822c38a92ec0a9
def deep_merge(a: dict, b: dict) -> dict:
    result = copy.deepcopy(a)
    for bk, bv in b.items():
        av = result.get(bk)
        if isinstance(av, dict) and isinstance(bv, dict):
            result[bk] = deep_merge(av, bv)
        else:
            result[bk] = copy.deepcopy(bv)
    return result


def load_config(colon_separated_filenames):
    filenames = colon_separated_filenames.split(":")
    config = {}
    for fname in filenames:
        with open(fname) as f:
            config = deep_merge(config, yaml.safe_load(f))
    return config
