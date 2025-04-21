__all__ = [
    'boolean',
    'CMAPS',
    'ClientSideError',
    'Color',
    'ColorScale',
    'InvalidRequestError',
    'NotFoundError',
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
    'AQUAMARINE',
    'BLACK',
    'BLUE',
    'BROWN',
    'CYAN',
    'DARKGREEN',
    'DARKORANGE',
    'DARKRED',
    'DEEPSKYBLUE',
    'FIREBRICK',
    'GREEN',
    'LIMEGREEN',
    'MOCCASIN',
    'NAVY',
    'ORANGE',
    'PALEGREEN',
    'PURPLE',
    'RED',
    'SIENNA',
    'TURQUOISE',
    'WHITE',
    'YELLOW',
]

import copy
import io
from typing import Tuple, List, Literal, Optional, Union, Callable, Iterable as Iterable
from typing import NamedTuple
import math
import datetime
import json
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
    """A class to define and manipulate colorscales.

    `ColorScale` instances are defined by colors at anchor points and are named.
    Colors are interpolated from one anchor to the next.
    To make a band of the same color, following anchors should be equal.
    To mark a discontinuity, following colors should differ and have the same anchor value.

    Parameters
    ----------
    name : str
        identifies the `ColorScale`
    colors : 1d-array[Colors]
        array of Colors class instances at anchor points
    scale : 1d-array[float], optional
        array of anchor points where `colors` are assigned.
        Most be monotically increasing and of same length as `colors`
        (default is from 0 to the number of `colors` minus 1 every 1).

    See Also
    --------
    Colors
    """
    
    def __init__(self, name, colors, scale=None):
        self.name = name
        self.colors = np.array(colors)
        if scale is None:
            self.scale = list(np.arange(len(colors)))
        else:
           if (np.diff(scale) < 0).any():
              raise Exception("scale must be monotically increasing")
           elif len(colors) == len(scale):
               self.scale = scale
           else:
               raise Exception("scale must be same length as colors")

    def reversed(self, name=None):
        """Reverts the order of the `colors` of a ColorScale instance.
        
        Parameters
        ----------
        name : str, optional
            a new name for the reversed ColorScale
            (default appends "_r" to the ColorScale's `name`)
        
        Returns
        -------
        ColorScale
            a renamed ColorScale of which `colors` 's order has been reversed.
        """
        if name is None:
            name = self.name + "_r"
        return ColorScale(name, self.colors[::-1], self.scale)

    def rescaled(self, new_min, new_max):
        """Rescales a ColorScale instance to new minimum and maximum.

        Parameters
        ----------
        new_min : float
            new minimum for the ColorScale
        new_max : float
            new maximum for the ColorScale. Must be greater then `new_min`

        Returns
        -------
        ColorScale
            a ColorScale of which anchors were rescaled from `new_min` to `new_max`
        """
        if new_max <= new_min:
            raise Exception("new_max must be greater than new_min")
        cs_val = np.array(self.scale)
        scale = (cs_val - cs_val[0]) * (new_max - new_min) / (cs_val[-1] - cs_val[0]) + new_min
        return ColorScale(self.name, self.colors, scale)

    def to_rgba_array(self, lutsize=256):
        """A `lutsize` -by-RGBA array representation of a ColorScale instance.

        Parameters
        ----------
        lutsize : int, optional
            the number of RGBA quantization levels (default is `lutsize` =256)

        Returns
        -------
        2d-array[int]
            `lutsize` RGBA quantizations linearly interpolated between anchors.

        See Also
        --------
        to_bgra_array 
        """
        cs = self.rescaled(0, lutsize-1)
        n_anchors =  len(cs.scale)
        # append output is not used but saves writing a condition dedicated to last color
        delta_colors  = np.diff(cs.colors, axis=0, append=np.expand_dims(cs.colors[-1,:], 0))
        delta_scale = np.diff(cs.scale, append=cs.scale[-1])
        # Construct lutsize x 4 Color array
        rgbaa = np.transpose(
            np.array(
                [
                    np.piecewise(
                        # Rescale lut indices to colors piece by piece
                        np.arange(lutsize),
                        # Rescale differently from one anchor point to the next
                        [(np.arange(lutsize) >= cs.scale[i])
                            & (np.arange(lutsize)  < cs.scale[i+1])
                                for i in range(n_anchors-1)]
                        + [np.arange(lutsize) == cs.scale[-1]],
                        # Rescaling is linear from one anchor to the next
                        [np.polynomial.polynomial.Polynomial(
                            # Unless it's a discontinuity then there is no rescaling
                            [cs.colors[i, band], 0] if delta_scale[i] == 0
                                # Intercept and slope of the linear relation
                                else [
                                    cs.colors[i, band] - cs.scale[i]
                                        * delta_colors[i, band] / delta_scale[i],
                                    delta_colors[i, band] / delta_scale[i],
                                ]
                        ) for i in range(n_anchors)]
                        # This is the lambda version of the Polynomial
                        # just can't get it to work... Leaving it here for someone smarter than me
                        ##[lambda x: (cs.colors[i, band]
                        ##    + 0 if np.diff(cs.scale, append=cs.scale[-1])[i] == 0 else ((x - cs.scale[i]) * delta_colors[i, band] / delta_scale[i])
                        ##) for i in range(n_anchors)]
                    # Same rescaling to all color bands
                    ) for band in range(4)
                ]
            )
        ).astype(int)
        return rgbaa

    def to_bgra_array(self, lutsize=256):
        """A `lutsize` -by-BGRA array representation of a ColorScale instance.

        Parameters
        ----------
        lutsize : int, optional
            the number of BGRA quantization levels (default is `lutsize` =256)

        Returns
        -------
        2d-array[int]
            `lutsize` BGRA quantizations linearly interpolated between anchors.

        See Also
        --------
        to_rgba_array 
        """
        return self.to_rgba_array(lutsize=lutsize)[:,[2, 1, 0, 3]]

    def to_dash_leaflet(self, lutsize=256):
        """A hexadecimal `lutsize` array representation of a ColorScale instance.

        Parameters
        ----------
        lutsize : int, optional
            the number of hexadecimal quantization levels (default is `lutsize` =256)

        Returns
        -------
        1d-array[hexadecimal color]
            `lutsize` hexadecimal colors quantizations linearly interpolated between anchors.
        
        See Also
        --------
        Color.to_hex_rgba
        """
        return [Color(*x).to_hex_rgba() for x in self.to_rgba_array(lutsize=lutsize)]


class Color(NamedTuple):
    """A sub-class of NamedTuple to define RGBA colors.
    
    Parameters
    ----------
    red : int
       intensity of red between 0 and 255
    green : int
       intensity of green between 0 and 255
    blue : int
       intensity of blue between 0 and 255
    alpha : int, optional
       transparency between 0 and 255 (default is `alpha` : 255)
    
    Example
    -------
    >>> DEEPSKYBLUE = Color(0, 191, 255) 
    (red: 0, green: 191, blue: 255, alpha: 255)
    """
    red: int
    green: int
    blue: int
    alpha: int = 255

    def to_hex_rgba(self):
        """Converts integer RGBA representation to hexadecimal RGBA
        
        Returns
        -------
        hexadecimal RGBA color

        Example
        -------
        >>> DEEPSKYBLUE = Color(0, 191, 255)
        >>> DEEPSKYBLUE.to_hex_rgba()
        #00bfffff
        """
        return f"#{self.red:02x}{self.green:02x}{self.blue:02x}{self.alpha:02x}"

    def to_hex_bgra(self):
        """Converts integer RGBA representation to hexadecimal BGRA
        
        Returns
        -------
        hexadecimal BGRA color

        Example
        -------
        >>> DEEPSKYBLUE = Color(0, 191, 255)
        >>> DEEPSKYBLUE.to_hex_rgba()
        #ffbf00ff
        """
        return f"#{self.blue:02x}{self.green:02x}{self.red:02x}{self.alpha:02x}"


class DrawAttrs(NamedTuple):
    line_color: Union[int, Color]
    background_color: Union[int, Color]
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
    im = apply_colormap(
        z,
        da.attrs["colormap"].to_bgra_array(lutsize=256),
        da.attrs["scale_min"],
        da.attrs["scale_max"],
    ) 
    if clipping is not None:
        if callable(clipping):
            clipping = clipping()
        draw_attrs = DrawAttrs(
            Color(255, 0, 0, 255), Color(0, 0, 0, 0), 1, cv2.LINE_AA
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
    color: Union[int, Color] = 255,
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
    fg_color: Union[int, Color] = 255,
    bg_color: Union[int, Color] = 0,
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
    im: np.ndarray, mask: np.ndarray, mask_color: Color = Color(0, 0, 0, 0)
) -> np.ndarray:
    h = im.shape[0]
    w = im.shape[1]
    mask = mask.reshape(mask.shape + (1,)).astype(np.float64) / 255
    mask_color = np.array(
        [mask_color.blue, mask_color.green, mask_color.red, mask_color.alpha],
        np.float64
    ).reshape((1, 1, 4))
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


AQUAMARINE = Color(127, 255, 212)
BLACK = Color(0, 0, 0)
BLUE = Color(0, 0, 255)
BROWN = Color(165, 42, 42)
CYAN = Color(0, 255, 255)
DARKGREEN = Color(0, 100, 0)
DARKORANGE = Color(255, 140, 0)
DARKRED = Color(128, 0, 0)
DEEPSKYBLUE = Color(0, 191, 255)
FIREBRICK = Color(178, 34, 34)
GREEN = Color(0, 255, 0)
LIMEGREEN = Color(50, 205, 50)
MOCCASIN = Color(255, 228, 181)
NAVY = Color(0, 0, 128)
ORANGE = Color(255, 165, 0)
PALEGREEN = Color(152, 251, 152)
PURPLE = Color(160, 32, 240)
RED = Color(255, 0, 0)
SIENNA = Color(160, 82, 45)
TURQUOISE = Color(64, 224, 208)
WHITE = Color(255, 255, 255)
YELLOW = Color(255, 255, 0)

_CORRELATION_CS = ColorScale(
    "correlation",
    [NAVY, BLUE, DEEPSKYBLUE, AQUAMARINE, PALEGREEN, MOCCASIN, MOCCASIN, YELLOW, DARKORANGE, RED, DARKRED],
    [-1, -0.8, -0.6, -0.3, -0.1, -0.1, 0.1, 0.1, 0.4, 0.7, 1],
)

_STD_ANOMALY_CS = ColorScale(
    "std_anomaly",
    [
        Color(0, 0, 110),
        Color(0, 0, 130),
        Color(0, 0, 130),
        Color(20, 20, 170),
        Color(20, 20, 170),
        Color(70, 70, 190),
        Color(70, 70, 190),
        Color(150, 150, 230),
        Color(150, 150, 230),
        Color(170, 170, 242),
        Color(170, 170, 242),
        Color(190, 190, 255),
        Color(190, 190, 255),
        Color(240, 240, 80),
        Color(240, 240, 80),
        Color(255, 255, 0),
        Color(255, 255, 0),
        Color(255, 215, 0),
        Color(255, 215, 0),
        Color(255, 165, 0),
        Color(255, 165, 0),
        Color(255, 99, 71),
        Color(255, 99, 71),
        Color(255, 0, 0),
        Color(255, 0, 0),
    ],
    [-3, -3, -2.5, -2.5, -2, -2, -1.5, -1.5, -1, -1, -0.5, -0.5, 0,
        0, 0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5, 3],
)

_RAINBOW_CS = ColorScale("rainbow", [
    Color(0, 0, 255),
    Color(0, 255, 255),
    Color(0, 255, 0),
    Color(255, 255, 0),
    Color(255, 0, 0),
    Color(255, 0, 255),
])

_PRECIP_CS = ColorScale(
    "precip",
    [
        WHITE,
        WHITE,
        Color(210, 255, 215),
        Color(210, 255, 215),
        Color(150, 230, 155),
        Color(150, 230, 155),
        Color(110, 210, 115),
        Color(110, 210, 115),
        Color(45, 180, 50),
        Color(45, 180, 50),
        Color(20, 170, 25),
        Color(20, 170, 25),
        Color(10, 150, 15),
        Color(10, 150, 15),
        Color(0, 130, 5),
        Color(0, 130, 5),
        Color(0, 110, 4),
        Color(0, 110, 4),
    ],
    [0, 0.2, 0.2, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16],
)

_PRCP_ANOMALY_CS = ColorScale(
    "prcp_anomaly",
    [
        Color(130, 90, 80),
        Color(130, 90, 80),
        Color(150, 110, 100),
        Color(150, 110, 100),
        Color(170, 130, 120),
        Color(170, 130, 120),
        Color(180, 140, 130),
        Color(180, 140, 130),
        Color(190, 150, 140),
        Color(190, 150, 140),
        Color(200, 160, 150),
        Color(200, 160, 150),
        Color(220, 185, 175),
        Color(220, 185, 175),
        Color(240, 215, 210),
        Color(240, 215, 210),
        WHITE,
        WHITE,
        Color(210, 255, 215),
        Color(210, 255, 215),
        Color(150, 230, 155),
        Color(150, 230, 155),
        Color(110, 210, 115),
        Color(110, 210, 115),
        Color(45, 180, 50),
        Color(45, 180, 50),
        Color(20, 170, 25),
        Color(20, 170, 25),
        Color(10, 150, 15),
        Color(10, 150, 15),
        Color(0, 130, 5),
        Color(0, 130, 5),
        Color(0, 110, 4),
        Color(0, 110, 4),
    ],
    [-16, -14, -14, -12, -12, -10, -10, -8, -8, -6, -6, -4, -4, -2, -2, -0.5, -0.5,
        0.5, 0.5, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16],
)

_PRCP_ANOMALY_BLUE_CS = ColorScale(
    "prcp_anomaly_blue",
    [
        SIENNA,
        SIENNA,
        Color(130, 90, 80),
        Color(130, 90, 80),
        Color(130, 90, 80),
        Color(130, 90, 80),
        Color(150, 110, 100),
        Color(150, 110, 100),
        Color(170, 130, 120),
        Color(170, 130, 120),
        Color(180, 140, 130),
        Color(180, 140, 130),
        Color(190, 150, 140),
        Color(190, 150, 140),
        Color(200, 160, 150),
        Color(200, 160, 150),
        Color(220, 185, 175),
        Color(220, 185, 175),
        Color(240, 215, 210),
        Color(240, 215, 210),
        Color(250, 235, 230),
        Color(250, 235, 230),
        WHITE,
        WHITE,
        Color(224, 255, 255),
        Color(224, 255, 255),
        Color(191, 239, 255),
        Color(191, 239, 255),
        Color(176, 226, 255),
        Color(176, 226, 255),
        Color(135, 206, 250),
        Color(135, 206, 250),
        Color(0, 191, 255),
        Color(0, 191, 255),
        Color(30, 144, 255),
        Color(30, 144, 255),
        Color(65, 105, 255),
        Color(65, 105, 255),
        Color(0, 0, 205),
        Color(0, 0, 205),
        Color(39, 64, 139),
        Color(39, 64, 139),
        Color(0, 0, 128),
        Color(0, 0, 128),
        Color(47, 47, 79),
        Color(47, 47, 79),
    ],
    [-500, -450, -450, -400, -400, -350, -350, -300, -300, -250, -250, -200, -200,
        -150, -150, -100, -100, -50, -50, -25, -25, -10, -10, 10, 10, 25, 25, 50, 50,
        100, 100, 150, 150, 200, 200, 250, 250, 300, 300, 350, 350, 400, 400,
        450, 450, 500],
)

_RAIN_POE_CS = ColorScale(
    "rain_poe",
    [BLACK, BROWN, ORANGE, YELLOW, MOCCASIN, MOCCASIN, LIMEGREEN, TURQUOISE, BLUE, PURPLE],
    [0, 0.15, 0.30, 0.45, 0.45, 0.55, 0.55, 0.7, 0.85, 1],
)

_TEMP_CS = ColorScale(
    "temp",
    [
        Color(0, 0, 110),
        Color(0, 0, 130),
        Color(0, 0, 130),
        Color(20, 20, 170),
        Color(20, 20, 170),
        Color(70, 70, 190),
        Color(70, 70, 190),
        Color(150, 150, 230),
        Color(150, 150, 230),
        Color(170, 170, 242),
        Color(170, 170, 242),
        Color(190, 190, 255),
        Color(190, 190, 255),
        Color(220, 220, 255),
        Color(220, 220, 255),
        Color(225, 245, 255),
        Color(225, 245, 255),
        Color(215, 255, 215),
        Color(215, 255, 215),
        Color(238, 232, 170),
        Color(238, 232, 170),
        Color(240, 240, 80),
        Color(240, 240, 80),
        Color(255, 255, 0),
        Color(255, 255, 0),
        Color(255, 215, 0),
        Color(255, 215, 0),
        Color(255, 165, 0),
        Color(255, 165, 0),
        Color(255, 99, 71),
        Color(255, 99, 71),
        Color(255, 00, 00),
        Color(255, 00, 00),
    ],
    [-40, -40, -35, -35, -30, -30, -25, -25, -20, -20, -15, -15, -10, -10, -5, -5, 0,
        0, 5, 5, 10, 10, 15, 15, 20, 20, 25, 25, 30, 30, 35, 35, 40],
)

_TEMP_ANOMALY_CS = ColorScale(
    "temp_anomaly",
    [PURPLE, CYAN, WHITE, WHITE, YELLOW, RED, FIREBRICK],
    [-10, -1, -1, 1, 1, 10, 10],
)

_RAIN_PNE_CS = _RAIN_POE_CS.reversed(name="rain_pne")

_VULN_CS = ColorScale(
    "vulnerability",
    [
        YELLOW,
        YELLOW,
        Color(255, 245, 0),
        Color(255, 245, 0),
        Color(255, 222, 0),
        Color(255, 222, 0),
        Color(255, 195, 0),
        Color(255, 195, 0),
        Color(255, 174, 0),
        Color(255, 174, 0),
        Color(255, 153, 0),
        Color(255, 153, 0),
        Color(255, 126, 0),
        Color(255, 126, 0),
        Color(255, 98, 0),
        Color(255, 98, 0),
        Color(255, 70, 0),
        Color(255, 70, 0),
        Color(255, 47, 0),
        Color(255, 47, 0),
        Color(255, 11, 0),
        Color(255, 11, 0),
        Color(247, 0, 0),
        Color(247, 0, 0),
        Color(219, 0, 0),
        Color(219, 0, 0),
        Color(191, 0, 0),
        Color(191, 0, 0),
        Color(166, 0, 0),
        Color(166, 0, 0),
        Color(128, 0, 0),
        Color(128, 0, 0),
    ],
    [0, 1, 1, 1.2, 1.2, 1.4, 1.4, 1.6, 1.6, 1.8, 1.8,
     2, 2, 2.2, 2.2, 2.4, 2.4, 2.6, 2.6, 2.8, 2.8,
     3, 3, 3.2, 3.2, 3.4, 3.4, 3.6, 3.6, 3.8, 3.8, 5],
)

_PNE_25_CS = ColorScale(
    "pne_25",
    [
       Color(0, 0, 186),
       Color(0, 0, 186),
       Color(0, 79, 255),
       Color(0, 79, 255),
       Color(62, 197, 245),
       Color(62, 197, 245),
       Color(127, 255, 212),
       Color(127, 255, 212),
       Color(151, 252, 0),
       Color(151, 252, 0),
       Color(216, 254, 0),
       Color(216, 254, 0),
       Color(255, 232, 0),
       Color(255, 232, 0),
       Color(255, 172, 0),
       Color(255, 172, 0),
       Color(255, 123, 0),
       Color(255, 123, 0),
       Color(255, 89, 0),
       Color(255, 89, 0),
       Color(255, 53, 0),
       Color(255, 53, 0),
       Color(255, 19, 0),
       Color(255, 19, 0),
       Color(239, 0, 0),
       Color(239, 0, 0),
       Color(209, 0, 0),
       Color(209, 0, 0),
       Color(176, 0, 0),
       Color(176, 0, 0),
       Color(128, 0, 0),
       Color(128, 0, 0),
    ],
    [0, 5, 5, 10, 10, 15, 15, 20, 20, 25, 25, 30, 30, 35, 35, 40, 40, 45, 45,
     50, 50, 55, 55, 60, 60, 65, 65, 70, 70, 75, 75, 100],
)


CMAPS = {CS.name : CS for CS in [
    _CORRELATION_CS,
    _PNE_25_CS,
    _PRECIP_CS,
    _PRCP_ANOMALY_CS,
    _PRCP_ANOMALY_BLUE_CS,
    _RAIN_PNE_CS,
    _RAIN_POE_CS,
    _RAINBOW_CS,
    _STD_ANOMALY_CS,
    _TEMP_CS,
    _TEMP_ANOMALY_CS,
    _VULN_CS,
]}


def parse_color(s: str) -> Color:
    v = int(s, 0)  # 0 tells int() to guess radix
    return Color(v >> 0 & 0xFF, v >> 8 & 0xFF, v >> 16 & 0xFF, 255)


def parse_color_item(vs: List[Color], s: str) -> List[Color]:
    if s == "null":
        rs = [Color(0, 0, 0, 0)]
    elif s[0] == "[":
        rs = [parse_color(s[1:])]
    elif s[-1] == "]":
        n = int(s[:-1])
        assert 1 < n <= 255 and len(vs) >= 2
        first = vs[-2]
        last = vs[-1]
        vs = vs[:-2]
        rs = [
            Color(
                first.red + (last.red - first.red) * i / n,
                first.green + (last.green - first.green) * i / n,
                first.blue + (last.blue - first.blue) * i / n,
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
        v = Color(*x)
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
    im[mask] = 0  # an arbitrary value that can be cast to int
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


def with_alpha(c: Color, alpha) -> Color:
    return Color(*c[:3], alpha)


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
    # This function assumes that the lat and lon coordinates are
    # evenly spaced, but when we combine DataArrays with different lat
    # and lon coordinates into a single Dataset, they can end up with
    # non-evenly-spaced coordinates, because the Dataset coordinates
    # are the union of the individual DataArray coordinates. Drop
    # empty coordinate values to get back to the original
    # evenly-spaced one. (TODO we really shouldn't be combining
    # variables with different resolutions into the same Dataset.)
    ds = ds.where(ds.notnull(), drop=True)

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

def parse_arg(name: str, conversion=str, default=REQUIRED, qstring: Optional[str] = None):
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


def deep_merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        result = copy.copy(a)
        for k, bv in b.items():
            av = result.get(k)
            result[k] = deep_merge(av, bv)
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"Lists of different lengths"
        result = [deep_merge(av, bv) for av, bv in zip(a, b)]
    else:
        result = copy.deepcopy(b)
    return result


def load_config(colon_separated_filenames):
    filenames = colon_separated_filenames.split(":")
    config = {}
    for fname in filenames:
        with open(fname) as f:
            config = deep_merge(config, yaml.safe_load(f))
    return config


def boolean(s):
    '''Validate a boolean querystring argument and convert it to bool.'''
    if s == 'true':
        return True
    if s == 'false':
        return False
    assert False, "Valid values are 'true' and 'false'"
