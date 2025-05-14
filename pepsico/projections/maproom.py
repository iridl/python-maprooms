import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import pingrid
from pingrid import CMAPS
from . import layout
import plotly.graph_objects as pgo
import xarray as xr
import pandas as pd
from dateutil.relativedelta import *
from globals_ import FLASK, GLOBAL_CONFIG
import app_calc as ac
import maproom_utilities as mru
import numpy as np


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
        variable = "pr"
        data = ac.read_data(scenario, model, variable, region)
        zoom = {"SAMER": 3, "US-CA": 4, "SASIA": 4, "Thailand": 5}
        return mru.initialize_map(data) + (zoom[region],)
    

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
        variable = "pr"
        data = ac.read_data(scenario, model, variable, region)
        initialization_cases = ["region"]
        return mru.picked_location(
            data, initialization_cases, click_lat_lng, latitude, longitude
        )


    def local_data(lat, lng, region, model, variable, start_month, end_month):
        model = [model] if model != "Multi-Model-Average" else [
            "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR","MRI-ESM2-0", "UKESM1-0-LL"
        ]
        data_ds = xr.concat([xr.Dataset({
            "histo" : ac.read_data(
                "historical", m, variable, region, unit_convert=True,
            ),
            "picontrol" : ac.read_data(
                "picontrol", m, variable, region, unit_convert=True,
            ),
            "ssp126" : ac.read_data(
                "ssp126", m, variable, region, unit_convert=True,
            ),
            "ssp370" : ac.read_data(
                "ssp370", m, variable, region, unit_convert=True,
            ),
            "ssp585" : ac.read_data(
                "ssp585", m, variable, region, unit_convert=True,
            ),
        }) for m in model], "M").assign_coords({"M": [m for m in model]})
        error_msg = None
        missing_ds = xr.Dataset()
        if any([var is None for var in data_ds.data_vars.values()]):
            #This is not supposed to happen:
            #would mean something happened to that data
            data_ds = missing_ds
            error_msg="Data missing for this model or variable"
        try:
            data_ds = pingrid.sel_snap(data_ds, lat, lng)
        except KeyError:
            data_ds = missing_ds
            error_msg="Grid box out of data domain"
        if error_msg == None :
            data_ds = ac.seasonal_data(data_ds, start_month, end_month)
        return data_ds, error_msg


    def plot_ts(ts, name, color, start_format, units):
        return pgo.Scatter(
            x=ts["T"].dt.strftime(mru.STD_TIME_FORMAT),
            y=ts.values,
            customdata=ts["seasons_ends"].dt.strftime("%B %Y"),
            hovertemplate=("%{x|"+start_format+"}%{customdata}: %{y:.2f} " + units),
            name=name,
            line=pgo.scatter.Line(color=color),
            connectgaps=False,
        )
    

    def add_period_shape(
        graph, data, start_year, end_year, fill_color, line_color, annotation
    ):
        return graph.add_vrect(
            x0=data["seasons_starts"].where(
                lambda x : (x.dt.year == int(start_year)), drop=True
            ).dt.strftime(mru.STD_TIME_FORMAT).values[0],
            #it's hard to believe this is how it is done
            x1=(
                pd.to_datetime(data["seasons_ends"].where(
                    lambda x : (x.dt.year == int(end_year)), drop=True
                ).dt.strftime(mru.STD_TIME_FORMAT).values[0]
            ) + relativedelta(months=+1)).strftime(mru.STD_TIME_FORMAT),
            fillcolor=fill_color,  opacity=0.2,
            line_color=line_color, line_width=3,
            layer="below",
            annotation_text=annotation, annotation_position="top left",
            #editable=True, #a reminder it might be the way to interact
        )


    @APP.callback(
        Output("btn_csv", "disabled"),
        Input("lat_input", "value"),
        Input("lng_input", "value"),
        Input("lat_input", "min"),
        Input("lng_input", "min"),
        Input("lat_input", "max"),
        Input("lng_input", "max"),
    )
    def invalid_button(lat, lng, lat_min, lng_min, lat_max, lng_max):
        return (
            lat < float(lat_min) or lat > float(lat_max)
            or lng < float(lng_min) or lng > float(lng_max)
        )


    @APP.callback(
        Output("download-dataframe-csv", "data"),
        Input("btn_csv", "n_clicks"),
        State("loc_marker", "position"),
        State("region", "value"),
        State("variable", "value"),
        State("start_month", "value"),
        State("end_month", "value"),
        prevent_initial_call=True,
    )
    def send_data_as_csv(
        n_clicks, marker_pos, region, variable, start_month, end_month,
    ):
        lat = marker_pos[0]
        lng = marker_pos[1]
        start_month = ac.strftimeb2int(start_month)
        end_month = ac.strftimeb2int(end_month)
        model = "Multi-Model-Average"
        data_ds, error_msg = local_data(
            lat, lng, region, model, variable, start_month, end_month
        )
        if error_msg == None :
            lng_units = "E" if (lng >= 0) else "W"
            lat_units = "N" if (lat >= 0) else "S"
            file_name = (
                f'{data_ds["histo"]["T"].dt.strftime("%b")[0].values}-'
                f'{data_ds["histo"]["seasons_ends"].dt.strftime("%b")[0].values}'
                f'_{variable}_{abs(lat)}{lat_units}_{abs(lng)}{lng_units}'
                f'.csv'
            )
            df = data_ds.to_dataframe()
            return dash.dcc.send_data_frame(df.to_csv, file_name)
        else :
            return None


    @APP.callback(
        Output("local_graph", "figure"),
        Input("loc_marker", "position"),
        Input("region", "value"),
        Input("submit_controls","n_clicks"),
        State("model", "value"),
        State("variable", "value"),
        State("start_month", "value"),
        State("end_month", "value"),
        State("start_year", "value"),
        State("end_year", "value"),
        State("start_year_ref", "value"),
        State("end_year_ref", "value"),
    )
    def local_plots(
        marker_pos,
        region,
        n_clicks,
        model,
        variable,
        start_month,
        end_month,
        start_year,
        end_year,
        start_year_ref,
        end_year_ref,
    ):
        lat = marker_pos[0]
        lng = marker_pos[1]
        start_month = ac.strftimeb2int(start_month)
        end_month = ac.strftimeb2int(end_month)
        data_ds, error_msg = local_data(
            lat, lng, region, model, variable, start_month, end_month
        )
        if error_msg != None :
            local_graph = pingrid.error_fig(error_msg)
        else :
            if (end_month < start_month) :
                start_format = "%b %Y - "
            else:
                start_format = "%b-"
            local_graph = pgo.Figure()
            data_color = {
                "histo": "blue", "picontrol": "green",
                "ssp126": "yellow", "ssp370": "orange", "ssp585": "red",
            }
            lng_units = "˚E" if (lng >= 0) else "˚W"
            lat_units = "˚N" if (lat >= 0) else "˚S"
            for var in data_ds.data_vars:
                local_graph.add_trace(plot_ts(
                    data_ds[var].mean("M", keep_attrs=True), var, data_color[var],
                    start_format, data_ds[var].attrs["units"]
                ))
            add_period_shape(
                local_graph,
                data_ds,
                start_year_ref,
                end_year_ref,
                "blue",
                "RoyalBlue",
                "reference period",
            )
            add_period_shape(
                local_graph,
                data_ds,
                start_year,
                end_year,
                "LightPink",
                "Crimson",
                "projected period",
            )
            local_graph.update_layout(
                xaxis_title="Time",
                yaxis_title=(
                    f'{data_ds["histo"].attrs["long_name"]} '
                    f'({data_ds["histo"].attrs["units"]})'
                ),
                title={
                    "text": (
                        f'{data_ds["histo"]["T"].dt.strftime("%b")[0].values}-'
                        f'{data_ds["histo"]["seasons_ends"].dt.strftime("%b")[0].values}'
                        f' {variable} seasonal average from model {model} '
                        f'at ({abs(lat)}{lat_units}, {abs(lng)}{lng_units})'
                    ),
                    "font": dict(size=14),
                },
                margin=dict(l=30, r=30, t=30, b=30),
            )
        return local_graph


    @APP.callback(
        Output("map_description", "children"),
        Input("submit_controls", "n_clicks"),
        State("scenario", "value"),
        State("model", "value"),
        State("variable", "value"),
        State("start_month", "value"),
        State("end_month", "value"),
        State("start_year", "value"),
        State("end_year", "value"),
        State("start_year_ref", "value"),
        State("end_year_ref", "value"),
    )
    def write_map_description(
        n_clicks,
        scenario,
        model,
        variable,
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
        Input("submit_controls","n_clicks"),
        State("scenario", "value"),
        State("model", "value"),
        State("variable", "value"),
        State("start_month", "value"),
        State("end_month", "value"),
        State("start_year", "value"),
        State("end_year", "value"),
        State("start_year_ref", "value"),
        State("end_year_ref", "value"),
    )
    def write_map_title(
        n_clicks,
        scenario,
        model,
        variable,
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


    APP.clientside_callback(
        """function(start_year, end_year, start_year_ref, end_year_ref) {
            if (start_year && end_year) {
                invalid_start_year = (start_year > end_year)
                invalid_end_year = invalid_start_year
            } else {
                invalid_start_year = !start_year
                invalid_end_year = !end_year
            }
            if (start_year_ref && end_year_ref) {
                invalid_start_year_ref = (start_year_ref > end_year_ref)
                invalid_end_year_ref = invalid_start_year_ref
            } else {
                invalid_start_year_ref = !start_year_ref
                invalid_end_year_ref = !end_year_ref
            }
            return [
                invalid_start_year, invalid_end_year,
                invalid_start_year_ref, invalid_end_year_ref,
                (
                    invalid_start_year || invalid_end_year
                    || invalid_start_year_ref || invalid_end_year_ref
                ),
            ]
        }
        """,
        Output("start_year", "invalid"),
        Output("end_year", "invalid"),
        Output("start_year_ref", "invalid"),
        Output("end_year_ref", "invalid"),
        Output("submit_controls", "disabled"),
        Input("start_year", "value"),
        Input("end_year", "value"),
        Input("start_year_ref", "value"),
        Input("end_year_ref", "value"),
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
        model = [model] if model != "Multi-Model-Average" else [
            "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR","MRI-ESM2-0", "UKESM1-0-LL"
        ]
        ref = xr.concat([
            ac.seasonal_data(
                ac.read_data("historical", m, variable, region, unit_convert=True),
                start_month, end_month,
                start_year=start_year_ref, end_year=end_year_ref,
            ).mean(dim="T", keep_attrs=True) for m in model
        ], "M").mean("M", keep_attrs=True)
        data = xr.concat([
            ac.seasonal_data(
                ac.read_data(scenario, m, variable, region, unit_convert=True),
                start_month, end_month,
                start_year=start_year, end_year=end_year,
            ).mean(dim="T", keep_attrs=True) for m in model
        ], "M").mean("M", keep_attrs=True)
        #Tedious way to make a subtraction only to keep attributes
        data = xr.apply_ufunc(
            np.subtract, data, ref, dask="allowed", keep_attrs="drop_conflicts",
        )
        if variable in ["hurs", "huss", "pr"]:
            data = 100. * data / ref
            data.attrs["units"] = "%"
        return data.rename({"X": "lon", "Y": "lat"})


    def map_attributes(data):
        variable = data.name
        if variable in ["tas", "tasmin", "tasmax"]:
            colorscale = CMAPS["temp_anomaly"]
        elif variable in ["hurs", "huss"]:
            colorscale = CMAPS["prcp_anomaly"].rescaled(-30, 30)
        elif variable in ["pr"]:
            colorscale = CMAPS["prcp_anomaly"].rescaled(-100, 100)
        else:
            map_amp = np.max(np.abs(data)).values
            if variable in ["prsn"]:
                colorscale = CMAPS["prcp_anomaly_blue"]
            elif variable in ["sfcwind"]:
                colorscale = CMAPS["std_anomaly"]
            else:
                colorscale = CMAPS["correlation"]
            colorscale = colorscale.rescaled(-1*map_amp, map_amp)
        return colorscale, colorscale.scale[0], colorscale.scale[-1]


    @APP.callback(
        Output("colorbar", "colorscale"),
        Output("colorbar", "min"),
        Output("colorbar", "max"),
        Output("colorbar", "unit"),
        Input("region", "value"),
        Input("submit_controls","n_clicks"),
        State("scenario", "value"),
        State("model", "value"),
        State("variable", "value"),
        State("start_month", "value"),
        State("end_month", "value"),
        State("start_year", "value"),
        State("end_year", "value"),
        State("start_year_ref", "value"),
        State("end_year_ref", "value"),
    )
    def draw_colorbar(
        region,
        n_clicks,
        scenario,
        model,
        variable,
        start_month,
        end_month,
        start_year,
        end_year,
        start_year_ref,
        end_year_ref,
    ):
        model = [model] if model != "Multi-Model-Average" else [
            "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR","MRI-ESM2-0", "UKESM1-0-LL"
        ]
        data = xr.concat([seasonal_change(
            scenario,
            m,
            variable,
            region,
            ac.strftimeb2int(start_month),
            ac.strftimeb2int(end_month),
            int(start_year),
            int(end_year),
            int(start_year_ref),
            int(end_year_ref),
        ) for m in model], "M").mean("M", keep_attrs=True)
        colorbar, min, max = map_attributes(data)
        return colorbar.to_dash_leaflet(), min, max, data.attrs["units"]


    @APP.callback(
        Output("layers_control", "children"),
        Output("map_warning", "is_open"),
        Input("region", "value"),
        Input("submit_controls", "n_clicks"),
        State("scenario", "value"),
        State("model", "value"),
        State("variable", "value"),
        State("start_month", "value"),
        State("end_month", "value"),
        State("start_year", "value"),
        State("end_year", "value"),
        State("start_year_ref", "value"),
        State("end_year_ref", "value"),
    )
    def make_map(
        region,
        n_clicks,
        scenario,
        model,
        variable,
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
                f"{TILE_PFX}/{{z}}/{{x}}/{{y}}/{region}/{scenario}/{model}/{variable}/"
                f"{start_month}/{end_month}/{start_year}/{end_year}/{start_year_ref}/"
                f"{end_year_ref}"
            )
        except:
            url_str= ""
            send_alarm = True
        return mru.layers_controls(
            url_str, f"change_{region}", "Change",
            GLOBAL_CONFIG["datasets"][f"shapes_adm_{region}"], adm_id_suffix=region,
        ), send_alarm


    @FLASK.route(
        (
            f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>/<region>/<scenario>/<model>/<variable>/"
            f"<start_month>/<end_month>/<start_year>/<end_year>/<start_year_ref>/"
            f"<end_year_ref>"
        ),
        endpoint=f"{config['core_path']}"
    )
    def fcst_tiles(tz, tx, ty,
        region,
        scenario,
        model,
        variable,
        start_month,
        end_month,
        start_year,
        end_year,
        start_year_ref,
        end_year_ref,
    ):
        # Reading\
        model = [model] if model != "Multi-Model-Average" else [
            "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR","MRI-ESM2-0", "UKESM1-0-LL"
        ]
        data = xr.concat([seasonal_change(
            scenario,
            m,
            variable,
            region,
            ac.strftimeb2int(start_month),
            ac.strftimeb2int(end_month),
            int(start_year),
            int(end_year),
            int(start_year_ref),
            int(end_year_ref),
        ) for m in model], "M").mean("M", keep_attrs=True)
        (
            data.attrs["colormap"],
            data.attrs["scale_min"],
            data.attrs["scale_max"],
        ) = map_attributes(data)
        resp = pingrid.tile(data, tx, ty, tz)
        return resp
