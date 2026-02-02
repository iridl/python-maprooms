import pingrid
import pandas as pd
import dash
import dash_leaflet as dlf
import psycopg2
from psycopg2 import sql
import shapely
from shapely import wkb
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry import Polygon


# Mapping Utilities

STD_TIME_FORMAT = "%Y-%m-%d"
HUMAN_TIME_FORMAT = "%-d %b %Y"


def get_geom(level, conf, shapes_adm_name):
    """ Form a geometric object from sql query or synthetic

    Parameters
    ----------
    level: int
        level from the enumeration of a suite of administrative boundaries listed in
        `conf` . Synthetic case limited to 0 and 1. 
    conf: dict
        dictionary listing desired administrative shapes and their attributes.

    Returns
    -------
    df : pandas.DataFrame
        a pd.DF with columns "label" (dtype=string),
        "key" (string or int depending on the table),
        and "the_geom" (shapely.Geometry)

    See Also
    --------
        synthesize_geom, sql2geom
    """
    if "bbox" in conf["datasets"] :
        return synthesize_geom(conf["datasets"]["bbox"], level=level)
    else:
        return sql2geom(conf["datasets"][shapes_adm_name][level]["sql"], conf["db"])


def sql2GeoJSON(shapes_sql, db_config):
    """ Form a GeoJSON dict from sql request to a database

    Parameters
    ----------
    shapes_sql: str
        sql request
    db_config: dict
        dictionary with host, port, user and dbname information
    
    Returns
    -------
    features: dict
        dictionary with features as key and GeoJSON of shapes_sql as value

    See Also
    --------
    sql2geom, geom2GeoJSON

    Examples
    --------
    shapes_sql: select id_1 as key, name_1 as label,
        ST_AsBinary(the_geom) as the_geom from sen_adm1
    db_config:
        host: postgres
        port: 5432
        user: ingrid
        dbname: iridb
    """
    return geom2GeoJSON(sql2geom(shapes_sql, db_config))


def geom2GeoJSON(df):
    """ Form a GeoJSON dict from a geometric object

    Parameters
    ----------
    df: geometric object
        shapely geometric object
    
    Returns
    -------
    features: dict
        dictionary with features as key and GeoJSON of `geom` as value

    See Also
    --------
    sql2geom, shapely.MultiPolygon, shapely.geometry.mapping
    """
    df["the_geom"] = df["the_geom"].apply(
        lambda x: x if isinstance(x, MultiPolygon) else MultiPolygon([x])
    )
    shapes = df["the_geom"].apply(shapely.geometry.mapping)
    for i in df.index: #this adds the district layer as a label in the dict
        shapes[i]['label'] = df['label'][i]
    return {"features": shapes}


def sql2geom(shapes_sql, db_config):
    """ Form a geometric object from sql query to a database

    Parameters
    ----------
    shapes_sql: str
        sql query
    db_config: dict
        dictionary with host, port, user and dbname information
    
    Returns
    -------
    df : pandas.DataFrame
        a pd.DF with columns "label" (dtype=string),
        "key" (string or int depending on the table),
        and "the_geom" (shapely.Geometry)

    See Also
    --------
    psycopg2.connect, psycopg2.sql, pandas.read_sql, shapely.wkb,

    Examples
    --------
    shapes_sql: select id_1 as key, name_1 as label,
        ST_AsBinary(the_geom) as the_geom from sen_adm1
    db_config:
        host: postgres
        port: 5432
        user: ingrid
        dbname: iridb
    """
    with psycopg2.connect(**db_config) as conn:
        s = sql.Composed(
            [
                sql.SQL("with g as ("),
                sql.SQL(shapes_sql),
                sql.SQL(
                    """
                    )
                    select
                        g.label, g.key, g.the_geom
                    from g
                    """
                ),
            ]
        )
        df = pd.read_sql(s, conn)
    df["the_geom"] = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))
    return df


def synthesize_geom(bbox, level):
    """ Synthesize a geometric object from a bounding box

    Parameters
    ----------
    bbox : array
        coordinates of bounding box of spatial domain as [W, S, E, N]
    level : int
        0 or 1 to mimick a containing admin level (0) with 1 geometry roughly smaller
        than `bbox` or a contained admin level (1) with 2 geometries partitioning
        level 0
    
    Returns
    -------
    df : pandas.DataFrame
        a pd.DF with columns "label" (dtype=string),
        "key" (string or int depending on the table),
        and "the_geom" (shapely.Geometry)

    See Also
    --------
    shapely.geometry.Polygon

    Notes
    -----
    A level 0 contained into bbox is necessary to test the clipping feature since
    `bbox` is also used to generate the fake data.
    """
    west, south, east, north = bbox
    assert (south + 0.25) <= (north - 0.5), (
        "Please extend latitudinal domain of bbox"
    )
    if east < west :
        assert (west + 0.25) >= (east - 0.5), (
            "Please extend longitudinal domain of bbox"
        )
    else :
        assert (west + 0.25) <= (east - 0.5), (
            "Please extend longitudinal domain of bbox"
        )
    west = west + 0.25
    south = south + 0.25
    east = east - 0.5
    norht = north - 0.5
    if level == 0 :
        df = pd.DataFrame({"label" : ["Guyane"], "key": [0], "the_geom": [Polygon([
            [west, south], [west, north], [east, north], [east, south]
        ])]})
    elif level == 1 : #2 triangles partitioning level-0 box at its SW-NE diagnonal
        df = pd.DataFrame({"label" : ["NW", "SE"], "key": [1, 2],"the_geom": [
            Polygon([[west, south], [west, north], [east, north]]),
            Polygon([[west, south], [east, north], [east, south]]),
        ]})
    else:
        raise Exception("level must be 0 or 1")
    return df


def make_adm_overlay(
    adm_name, adm_id, adm_geojson, adm_color, adm_lev, adm_weight, is_checked=False
):
    """ Draw a dlf.Overlay of a dlf.GeoJSON for Maprooms admin

    Parameters
    ----------
    adm_name: str
        name to give the dlf.Overlay
    adm_id: str
        unique id of dlf.Overlay
    adm_geojson: dict
        GeoJSON of the admin
    adm_color: str
        color to give the dlf.GeoJSON
    adm_lev: int
        index to give the dlf.GeoJSON's id
    adm_weight: int
        weight to give the dlf.GeoJSON
    is_checked: boolean, optional
        whether dlf.Overlay is checked or not (default: not)

    Returns
    -------
    adm_layer : dlf.Overlay
        dlf.GeoJSON overlay with the given parameters

    See Also
    --------
    calc.sql2GeoJSON for the format of `adm_geojson`,
    dash_leaflet.Overlay, dash_leaflet.GeoJSON
    """
    border_id = {"type": "borders_adm", "index": adm_lev}
    return dlf.Overlay(
        dlf.GeoJSON(
            id=border_id,
            data=adm_geojson,
            options={
                "fill": True,
                "color": adm_color,
                "weight": adm_weight,
                "fillOpacity": 0,
            },
        ),
        name=adm_name,
        id=adm_id,
        checked=is_checked,
    )


def initialize_map(data):
    """
    Map initialization based on `data` spatial domain

    Parameters
    ----------
    data: xr.DataArray
        spatial data of which longitude and latitude coordinates are X and Y

    Returns
    -------
    lat_min, lat_max, lat_label, lon_min, lon_max, lon_label, center_of_the_map : tuple
        respectively: minimum, maximum, label for latitude and longitude pick a point
        control, center of the map coordinates values as a list
    """
    center_of_the_map = [
        ((data["Y"][int(data["Y"].size/2)].values)),
        ((data["X"][int(data["X"].size/2)].values)),
    ]
    lat_res = (data["Y"][0 ]- data["Y"][1]).values
    lat_min = str((data["Y"][-1] - lat_res/2).values)
    lat_max = str((data["Y"][0] + lat_res/2).values)
    lon_res = (data["X"][1] - data["X"][0]).values
    lon_min = str((data["X"][0] - lon_res/2).values)
    lon_max = str((data["X"][-1] + lon_res/2).values)
    lat_label = lat_min + " to " + lat_max + " by " + str(lat_res) + "˚"
    lon_label = lon_min + " to " + lon_max + " by " + str(lon_res) + "˚"
    return (
        lat_min, lat_max, lat_label,
        lon_min, lon_max, lon_label,
        center_of_the_map
    )


def picked_location(
    data, initialization_cases, click_lat_lng, latitude, longitude
):
    """
    Inputs for map loc_marker and pick location lat/lon controls

    Parameters
    ----------
    data: xr.DataArray
        spatial data of which longitude and latitude coordinates are X and Y
    initialization_cases: list[str]
        list of Input of which changes reinitialize the map
    click_lat_lng: list[str]
        dlf Input from clicking map (lat and lon)
    latitude: str
        Input from latitude pick a point control
    latitude: str
        Input from latitude pick a point control
    """
    if (
        dash.ctx.triggered_id == None
        or dash.ctx.triggered_id in initialization_cases
    ):
        lat = data["Y"][int(data["Y"].size/2)].values
        lng = data["X"][int(data["X"].size/2)].values
    else:
        if dash.ctx.triggered_id == "map":
            lat = click_lat_lng[0]
            lng = click_lat_lng[1]
        else:
            lat = latitude
            lng = longitude
        try:
            nearest_grid = pingrid.sel_snap(data, lat, lng)
            lat = nearest_grid["Y"].values
            lng = nearest_grid["X"].values
        except KeyError:
            lat = lat
            lng = lng
    return [lat, lng], lat, lng


def layers_controls(
    url_str, url_id, url_name,
    adm_conf, global_conf, adm_id_suffix=None,
    street=True, topo=True,
):
    """
    Input to dbf.LayersControl

    Parameters
    ----------
    url_str: str
        dlf.TileLayer's url
    url_id: str
        dlf.Overlay's id where dlf.TileLayer is
    url_name: str
        dlf.Overlay's name where dlf.TileLayer is
    adm_conf: dict
        configuration of administrative boundaries overlays
    global_conf: dict
        configuration of maproom app
    adm_id_suffix: str, optional
        suffix in `adm_conf` dictionary that identifies the set of admin to use
    street: boolean, optional
        use cartodb street map as dlf.BaseLayer
    topo: boolean, optional
        use opentopomap topographic map as dlf.BaseLayer

    Returns
    -------
    layers: list
        list of dlf.BaseLayer and dlf.Overlay
    """
    if adm_id_suffix is None:
        adm_id_suffix = ""
    layers = [
        make_adm_overlay(
            adm_name=adm["name"],
            adm_id=f'{adm["name"]}{adm_id_suffix}',
            adm_geojson=geom2GeoJSON(get_geom(
                level=i,
                conf=global_conf,
                shapes_adm_name=f'shapes_adm{adm_id_suffix}',
            )),
            adm_color=adm["color"],
            adm_lev=i+1,
            adm_weight=len(adm_conf)-i,
            is_checked=adm["is_checked"],
        )
        for i, adm in enumerate(adm_conf)
    ] + [
        dlf.Overlay(
            dlf.TileLayer(url=url_str, opacity=1),
            id=url_id,
            name=url_name,
            checked=True,
        ),
    ]
    if topo:
        layers = [
            dlf.BaseLayer(
                dlf.TileLayer(
                    url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
                ),
                name="Topo",
                checked=True,
            ),
        ] + layers
    if street:
        layers = [
            dlf.BaseLayer(
                dlf.TileLayer(
                    url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
                ),
                name="Street",
                checked=False,
            ),
        ] + layers
    return layers
