import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import pingrid
from pingrid import CMAPS
from . import layout
import plotly.graph_objects as pgo
import xarray as xr
import pandas as pd
import dash_leaflet as dlf
import psycopg2
from psycopg2 import sql
import shapely
from shapely import wkb
from shapely.geometry.multipolygon import MultiPolygon
from globals_ import FLASK, GLOBAL_CONFIG
import app_calc as ac
import numpy as np
#from fieldsets import Block, PickPoint


STD_TIME_FORMAT = "%Y-%m-%d"
HUMAN_TIME_FORMAT = "%-d %b %Y"

def register(FLASK, config):
    PFX = f"{GLOBAL_CONFIG['url_path_prefix']}/{config['core_path']}"
    TILE_PFX = f"{PFX}/tile"

    # App

    APP = dash.Dash(
        __name__,
        server=FLASK,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
        ],
        url_base_pathname=f"{PFX}/",
        meta_tags=[
            {"name": "description", "content": "Forecast"},
            {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
        ],
    )
    APP.title = "Forecast"

    APP.layout = layout.app_layout()

    def adm_borders(shapes):
        with psycopg2.connect(**GLOBAL_CONFIG["db"]) as conn:
            s = sql.Composed(
                [
                    sql.SQL("with g as ("),
                    sql.SQL(shapes),
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
        df["the_geom"] = df["the_geom"].apply(
            lambda x: x if isinstance(x, MultiPolygon) else MultiPolygon([x])
        )
        shapes = df["the_geom"].apply(shapely.geometry.mapping)
        for i in df.index: #this adds the district layer as a label in the dict
            shapes[i]['label'] = df['label'][i]
        return {"features": shapes}


    def make_adm_overlay(
        adm_name, adm_sql, adm_color, adm_lev, adm_weight, is_checked=False,
    ):
        border_id = {"type": "borders_adm", "index": adm_lev}
        return dlf.Overlay(
            dlf.GeoJSON(
                id=border_id,
                data=adm_borders(adm_sql),
                options={
                    "fill": True,
                    "color": adm_color,
                    "weight": adm_weight,
                    "fillOpacity": 0
                },
            ),
            name=adm_name,
            checked=is_checked,
        )


    @APP.callback(
        Output("lat_input", "min"),
        Output("lat_input", "max"),
        Output("lat_input_tooltip", "children"),
        Output("lng_input", "min"),
        Output("lng_input", "max"),
        Output("lng_input_tooltip", "children"),
        Output("map", "center"),
        Output("map", "zoom"),
        Input("region", "value"),
        Input("location", "pathname"),
    )
    def initialize(region, path):
        scenario = "ssp126"
        model = "GFDL-ESM4"
        variable = "tasmin"
        data = ac.read_data(scenario, model, variable, region)
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
        zoom = {"SAMER": 3, "US-CA": 4, "SASIA": 4, "Thailand": 5}
        return (
            lat_min, lat_max, lat_label, lon_min, lon_max, lon_label,
            center_of_the_map, zoom[region],
        )


    @APP.callback(
        Output("loc_marker", "position"),
        Output("lat_input", "value"),
        Output("lng_input", "value"),
        Input("submit_lat_lng","n_clicks"),
        Input("map", "click_lat_lng"),
        Input("region", "value"),
        State("lat_input", "value"),
        State("lng_input", "value"),
    )
    def pick_location(n_clicks, click_lat_lng, region, latitude, longitude):
        # Reading
        scenario = "ssp126"
        model = "GFDL-ESM4"
        variable = "tasmin"
        data = ac.read_data(scenario, model, variable, region)
        if (dash.ctx.triggered_id == None or dash.ctx.triggered_id == "region"):
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


    def plot_ts(ts, name, color, start_format, units):
        return pgo.Scatter(
            x=ts["T"].dt.strftime(STD_TIME_FORMAT),
            y=ts.values,
            customdata=ts["seasons_ends"].dt.strftime("%B %Y"),
            hovertemplate=("%{x|"+start_format+"}%{customdata}: %{y:.2f}" + units),
            name=name,
            line=pgo.scatter.Line(color=color),
            connectgaps=False,
        )


    @APP.callback(
        Output("local_graph", "figure"),
        Input("loc_marker", "position"),
        Input("model", "value"),
        Input("variable", "value"),
        Input("region", "value"),
        Input("submit_season","n_clicks"),
        State("start_month", "value"),
        State("end_month", "value"),
    )
    def local_plots(marker_pos, model, variable, region, n_clicks, start_month, end_month):
        lat = marker_pos[0]
        lng = marker_pos[1]
        start_month = ac.strftimeb2int(start_month)
        end_month = ac.strftimeb2int(end_month)
        histo = ac.read_data("historical", model, variable, region)
        picont = ac.read_data("picontrol", model, variable, region)
        ssp126 = ac.read_data("ssp126", model, variable, region)
        ssp370 = ac.read_data("ssp370", model, variable, region)
        ssp585 = ac.read_data("ssp585", model, variable, region)
        # Should I make this a xr.ds?
        data_list = [histo, picont, ssp126, ssp370, ssp585]
        try:
            if any([var is None for var in data_list]):
                return pingrid.error_fig(
                    error_msg="Data missing for this model or variable"
                )
            else:
                data_list[:] = [pingrid.sel_snap(var, lat, lng) for var in data_list]
                if any([np.isnan(var).any() for var in data_list]):
                    return pingrid.error_fig(
                        error_msg="Data missing at this location"
                    )
        except KerError:
            return pingrid.error_fig(error_msg="Grid box out of data domain")

        data_list[:] = [
            ac.unit_conversion(ac.seasonal_data(var, start_month, end_month))
            for var in data_list
        ]
        if (end_month < start_month) :
            start_format = "%b %Y - "
        else:
            start_format = "%b-"
        if data_list[0].attrs["units"] == "Celsius" :
            units = "˚C"
        else:
            units = data_list[0].attrs["units"]
        local_graph = pgo.Figure()
        local_graph.add_trace(plot_ts(
            data_list[0], "histo", "blue", start_format, units
        ))
        local_graph.add_trace(plot_ts(
            data_list[1], "picontrol", "green", start_format, units
        ))
        local_graph.add_trace(plot_ts(
            data_list[2], "ssp126", "yellow", start_format, units
        ))
        local_graph.add_trace(plot_ts(
            data_list[3], "ssp370", "orange", start_format, units
        ))
        local_graph.add_trace(plot_ts(
            data_list[4], "ssp585", "red", start_format, units
        ))
        local_graph.update_layout(
            xaxis_title="Time",
            yaxis_title=f'{data_list[0].attrs["long_name"]} ({units})',
            title={
                "text": (
                    f'{data_list[0]["T"].dt.strftime("%b")[0].values}-'
                    f'{data_list[0]["seasons_ends"].dt.strftime("%b")[0].values} '
                    f'{variable} seasonal average from model {model}'
                ),
                "font": dict(size=14),
            }
        )
        return local_graph


    @APP.callback(
        Output("map_description", "children"),
        Input("scenario", "value"),
        Input("model", "value"),
        Input("variable", "value"),
        Input("submit_season", "n_clicks"),
        Input("submit_projy", "n_clicks"),
        Input("submit_refy", "n_clicks"),
        State("start_month", "value"),
        State("end_month", "value"),
        State("start_year", "value"),
        State("end_year", "value"),
        State("start_year_ref", "value"),
        State("end_year_ref", "value"),
    )
    def write_map_description(
        scenario,
        model,
        variable,
        n_clicks,
        py_clicks,
        ry_clicks,
        start_month,
        end_month,
        start_year,
        end_year,
        start_year_ref,
        end_year_ref,
    ):
        return (
            f'The Map displays the change in {start_month}-{end_month} seasonal average of '
            f'{variable} from {model} model under {scenario} scenario projected for '
            f'{start_year}-{end_year} with respect to historical {start_year_ref}-'
            f'{end_year_ref}'
        )


    @APP.callback(
        Output("map_title", "children"),
        Input("scenario", "value"),
        Input("model", "value"),
        Input("variable", "value"),
        Input("submit_season","n_clicks"),
        Input("submit_projy","n_clicks"),
        Input("submit_refy","n_clicks"),
        State("start_month", "value"),
        State("end_month", "value"),
        State("start_year", "value"),
        State("end_year", "value"),
        State("start_year_ref", "value"),
        State("end_year_ref", "value"),
    )
    def write_map_title(
        scenario,
        model,
        variable,
        n_clicks,
        py_clicks,
        ry_clicks,
        start_month,
        end_month,
        start_year,
        end_year,
        start_year_ref,
        end_year_ref,
    ):
        return (
            f'{start_month}-{end_month} {start_year}-{end_year} '
            f'{scenario} {model} {variable} change from '
            f'{start_year_ref}-{end_year_ref}'
        )


    def seasonal_change(
        scenario,
        model,
        variable,
        region,
        start_month,
        end_month,
        start_year,
        end_year,
        start_year_ref,
        end_year_ref,
    ):
        ref = ac.unit_conversion(ac.seasonal_data(
            ac.read_data("historical", model, variable, region),
            start_month, end_month,
            start_year=str(start_year_ref), end_year=str(end_year_ref),
        ).mean(dim="T"))
        data = ac.unit_conversion(ac.seasonal_data(
            ac.read_data(scenario, model, variable, region),
            start_month, end_month,
            start_year=str(start_year), end_year=str(end_year),
        ).mean(dim="T"))
        data = data - ref
        if variable in ["hurs", "huss", "pr"]:
            data = 100. * data / ref
            data["units"] = "%"
        return data.rename({"X": "lon", "Y": "lat"})


    def map_attributes(variable, data=None):
        if variable in ["tas", "tasmin", "tasmax"]:
            map_min = -8
            map_max = 8
        elif variable in ["hurs", "huss", "pr"]:
            map_min = -100
            map_max = 100
        else:
            assert (data is not None)
            map_amp = max(abs(data.min().values), abs(data.min().values))
            map_min = -1*map_amp
            map_max = map_amp
        return CMAPS["correlation"].rescaled(map_min, map_max), map_min, map_max


    @APP.callback(
        Output("colorbar", "colorscale"),
        Output("colorbar", "min"),
        Output("colorbar", "max"),
        Input("scenario", "value"),
        Input("model", "value"),
        Input("variable", "value"),
        Input("region", "value"),
        Input("submit_season","n_clicks"),
        Input("submit_projy","n_clicks"),
        Input("submit_refy","n_clicks"),
        State("start_month", "value"),
        State("end_month", "value"),
        State("start_year", "value"),
        State("end_year", "value"),
        State("start_year_ref", "value"),
        State("end_year_ref", "value"),
    )
    def draw_colorbar(
        scenario,
        model,
        variable,
        region,
        n_clicks,
        py_clicks,
        ry_clicks,
        start_month,
        end_month,
        start_year,
        end_year,
        start_year_ref,
        end_year_ref,
    ):
        if variable in ["tas", "tasmin", "tasmax", "hurs", "huss", "pr"]:
            colorscale, map_min, map_max = map_attributes(variable)
        else:
            start_month = ac.strftimeb2int(start_month)
            end_month = ac.strftimeb2int(end_month)
            data = seasonal_change(
            scenario,
            model,
            variable,
            region,
            start_month,
            end_month,
            start_year,
            end_year,
            start_year_ref,
            end_year_ref,
        )
            colorscale, map_min, map_max = map_attributes(variable, data=data)
        return colorscale.to_dash_leaflet(), map_min, map_max


    @APP.callback(
        Output("layers_control", "children"),
        Output("map_warning", "is_open"),
        Input("scenario", "value"),
        Input("model", "value"),
        Input("variable", "value"),
        Input("region", "value"),
        Input("submit_season","n_clicks"),
        Input("submit_projy","n_clicks"),
        Input("submit_refy","n_clicks"),
        State("start_month", "value"),
        State("end_month", "value"),
        State("start_year", "value"),
        State("end_year", "value"),
        State("start_year_ref", "value"),
        State("end_year_ref", "value"),
    )
    def make_map(
        scenario,
        model,
        variable,
        region,
        n_clicks,
        py_clicks,
        ry_clicks,
        start_month,
        end_month,
        start_year,
        end_year,
        start_year_ref,
        end_year_ref,
    ):
        try:
            send_alarm = False
            url_str = (
                f"{TILE_PFX}/{{z}}/{{x}}/{{y}}/{scenario}/{model}/{variable}/{region}/"
                f"{start_month}/{end_month}/{start_year}/{end_year}/{start_year_ref}/"
                f"{end_year_ref}"
            )
        except:
            url_str= ""
            send_alarm = True
        return [
            dlf.BaseLayer(
                dlf.TileLayer(
                    url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
                ),
                name="Street",
                checked=False,
            ),
            dlf.BaseLayer(
                dlf.TileLayer(
                    url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
                ),
                name="Topo",
                checked=True,
            ),
        ] + [
            make_adm_overlay(
                adm["name"],
                adm["sql"],
                adm["color"],
                i+1,
                len(GLOBAL_CONFIG["datasets"]["shapes_adm"])-i,
                is_checked=adm["is_checked"]
            )
            for i, adm in enumerate(GLOBAL_CONFIG["datasets"]["shapes_adm"])
        ] + [
            dlf.Overlay(
                dlf.TileLayer(url=url_str, opacity=1),
                name="Change",
                checked=True,
            ),
        ], send_alarm


    @FLASK.route(
        (
            f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>/<scenario>/<model>/<variable>/<region>/"
            f"<start_month>/<end_month>/<start_year>/<end_year>/<start_year_ref>/"
            f"<end_year_ref>"
        ),
        endpoint=f"{config['core_path']}"
    )
    def fcst_tiles(tz, tx, ty,
        scenario,
        model,
        variable,
        region,
        start_month,
        end_month,
        start_year,
        end_year,
        start_year_ref,
        end_year_ref,
    ):
        # Reading
        start_month = ac.strftimeb2int(start_month)
        end_month = ac.strftimeb2int(end_month)
        data = seasonal_change(
            scenario,
            model,
            variable,
            region,
            start_month,
            end_month,
            start_year,
            end_year,
            start_year_ref,
            end_year_ref,
        )
        (
            data.attrs["colormap"], data.attrs["scale_min"], data.attrs["scale_max"]
        ) = map_attributes(variable, data=data)
        resp = pingrid.tile(data, tx, ty, tz)
        return resp
