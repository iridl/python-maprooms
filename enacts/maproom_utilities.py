import dash_leaflet as dlf


# Mapping Utilities

def make_adm_overlay(
    adm_name, adm_geojson, adm_color, adm_lev, adm_weight, is_checked=False
):
    """ Draw a dlf.Overlay of a dlf.GeoJSON for Maprooms admin

    Parameters
    ----------
    adm_name: str
        name to give the dlf.Overlay
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
        checked=is_checked,
    )
