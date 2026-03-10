import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import pingrid
from pingrid import CMAPS
from . import layout  # Layout de la app definido en otro módulo
from . import extrafunctions  
import plotly.graph_objects as pgo
import xarray as xr  # Manejo de datos multidimensionales
import pandas as pd
from dateutil.relativedelta import *
from globals_ import FLASK, GLOBAL_CONFIG  # Configuraciones globales
import app_calc as ac  # Funciones de lectura y procesamiento de datos
import maproom_utilities as mru  # Funciones de utilidades para mapas
import numpy as np
#from dash_extensions.enrich import html

import math
import json
from dash_extensions.enrich import html
import dash_leaflet.express as dlx


from dash import dcc

import dash_leaflet as dlf


def register(FLASK, config):
    # Prefijos de URL para la app y los tiles
    PFX = f"{GLOBAL_CONFIG['url_path_prefix']}/{config['core_path']}"
    TILE_PFX = f"{PFX}/tile"

    # -------------------------------------------------
    # Inicialización de la app Dash
    # -------------------------------------------------
    APP = dash.Dash(
        __name__,
        server=FLASK,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,  # Uso del tema Bootstrap para la UI
        ],
        url_base_pathname=f"{PFX}/",  # Base URL de la app
        meta_tags=[  # Meta tags para responsividad y SEO
            {"name": "description", "content": "Forecast"},
            {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
        ],
    )

    APP.enable_dev_tools(debug=True)
    APP.title = "Forecast"  # Título de la pestaña del navegador

    # Definición del layout de la app
    APP.layout = layout.app_layout()

    # -------------------------------------------------
    # Callback para inicializar los inputs de lat/lng y el mapa según región
    # -------------------------------------------------
    # @APP.callback(
    #     Output("lat_input", "min"),
    #     Output("lat_input", "max"),
    #     Output("lat_input_tooltip", "children"),
    #     Output("lng_input", "min"),
    #     Output("lng_input", "max"),
    #     Output("lng_input_tooltip", "children"),
    #     Output("map", "center"),
    #     Output("map", "zoom"),
    #     Input("region", "value"),
    #     Input("location", "pathname"),
    # )
    # def initialize(region, path):
    #     # Valores por defecto
    #     scenario = "ssp126"
    #     model = "GFDL-ESM4"
    #     variable = "pr"

    #     # Lectura de datos usando app_calc
    #     data = ac.read_data(scenario, model, variable, region)

    #     # Diccionario de zoom inicial por región
    #     zoom = {"SAMER": 3, "US-CA": 4, "SASIA": 4, "Thailand": 5}

    #     # Inicialización de mapa y retorno de todos los valores necesarios para los inputs y el mapa
    #     return mru.initialize_map(data) + (zoom[region],)
    

    # -------------------------------------------------
    # Callback para seleccionar ubicación en el mapa
    # -------------------------------------------------
 #   @APP.callback(
    #     Output("loc_marker", "position"),  # Actualiza la posición del marcador
    #     Output("lat_input", "value"),  # Actualiza el input de latitud
    #     Output("lng_input", "value"),  # Actualiza el input de longitud
    #     Input("submit_lat_lng","n_clicks"),  # Evento de submit
    #     Input("map", "click_lat_lng"),  # Evento de click en el mapa
    #     Input("region", "value"),  # Región seleccionada
    #     State("lat_input", "value"),  # Latitud actual
    #     State("lng_input", "value"),  # Longitud actual
    # )
    # def pick_location(n_clicks, click_lat_lng, region, latitude, longitude):
    #     # Valores por defecto
    #     scenario = "ssp126"
    #     model = "GFDL-ESM4"
    #     variable = "pr"

    #     # Lectura de datos
    #     data = ac.read_data(scenario, model, variable, region)

    #     # Casos de inicialización para la función de utilidades
    #     initialization_cases = ["region"]

    #     # Retorna nueva posición y actualiza inputs usando utilidades
    #     return mru.picked_location(
    #         data, initialization_cases, click_lat_lng, latitude, longitude
    #     )


    # -------------------------------------------------
    # Función para extraer datos locales de un modelo y variable
    # -------------------------------------------------
    # def local_data(lat, lng, region, model, variable, start_month, end_month):
    #     # Si se selecciona Multi-Model-Average, se usan todos los modelos disponibles
    #     model = [model] if model != "Multi-Model-Average" else [
    #         "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR","MRI-ESM2-0", "UKESM1-0-LL"
    #     ]

    #     # Concatenación de datasets por modelo y escenario
    #     data_ds = xr.concat([xr.Dataset({
    #         "histo" : ac.read_data(
    #             "historical", m, variable, region, unit_convert=True,
    #         ),
    #         "picontrol" : ac.read_data(
    #             "picontrol", m, variable, region, unit_convert=True,
    #         ),
    #         "ssp126" : ac.read_data(
    #             "ssp126", m, variable, region, unit_convert=True,
    #         ),
    #         "ssp370" : ac.read_data(
    #             "ssp370", m, variable, region, unit_convert=True,
    #         ),
    #         "ssp585" : ac.read_data(
    #             "ssp585", m, variable, region, unit_convert=True,
    #         ),
    #     }) for m in model], "M").assign_coords({"M": [m for m in model]})

    #     # Manejo de errores
    #     error_msg = None
    #     missing_ds = xr.Dataset()

    #     # Verifica si hay datos faltantes
    #     if any([var is None for var in data_ds.data_vars.values()]):
    #         data_ds = missing_ds
    #         error_msg="Data missing for this model or variable"

    #     # Selecciona la grilla más cercana a lat/lng
    #     try:
    #         data_ds = pingrid.sel_snap(data_ds, lat, lng)
    #     except KeyError:
    #         data_ds = missing_ds
    #         error_msg="Grid box out of data domain"

    #     # Si no hubo errores, calcula los datos estacionales
    #     if error_msg == None :
    #         data_ds = ac.seasonal_data(data_ds, start_month, end_month)

    #     # Retorna dataset filtrado y posible mensaje de error
    #     return data_ds, error_msg
    # -------------------------------------------------
    # Función para generar un gráfico de serie de tiempo con Plotly
    # -------------------------------------------------
    def plot_ts(ts, name, color, start_format, units):
        return pgo.Scatter(
            # Eje X: fechas formateadas con el formato estándar definido en utilidades
            x=ts["T"].dt.strftime(mru.STD_TIME_FORMAT),
            # Eje Y: valores de la serie
            y=ts.values,
            # Datos personalizados para el hover: final de cada temporada
            customdata=ts["seasons_ends"].dt.strftime("%B %Y"),
            # Plantilla de hover con formato de fecha y unidades
            hovertemplate=("%{x|"+start_format+"}%{customdata}: %{y:.2f} " + units),
            name=name,
            # Color de la línea de la serie
            line=pgo.scatter.Line(color=color),
            # No conectar gaps
            connectgaps=False,
        )
    

    # -------------------------------------------------
    # Función para agregar un rectángulo vertical indicando un periodo en el gráfico
    # -------------------------------------------------
    def add_period_shape(
        graph, data, start_year, end_year, fill_color, line_color, annotation
    ):
        return graph.add_vrect(
            # Inicio del rectángulo: primer valor del año de inicio
            x0=data["seasons_starts"].where(
                lambda x : (x.dt.year == int(start_year)), drop=True
            ).dt.strftime(mru.STD_TIME_FORMAT).values[0],
            # Fin del rectángulo: último valor del año de fin + 1 mes
            x1=(
                pd.to_datetime(data["seasons_ends"].where(
                    lambda x : (x.dt.year == int(end_year)), drop=True
                ).dt.strftime(mru.STD_TIME_FORMAT).values[0]
            ) + relativedelta(months=+1)).strftime(mru.STD_TIME_FORMAT),
            # Estilo del rectángulo
            fillcolor=fill_color,  opacity=0.2,
            line_color=line_color, line_width=3,
            layer="below",  # Se dibuja debajo de las series
            annotation_text=annotation, annotation_position="top left",
            #editable=True, # Recordatorio de posible interacción futura
        )


    # -------------------------------------------------
    # Callback para habilitar o deshabilitar botón CSV según lat/lng válidos
    # -------------------------------------------------
    # @APP.callback(
    #     Output("btn_csv", "disabled"),
    #     Input("lat_input", "value"),
    #     Input("lng_input", "value"),
    #     Input("lat_input", "min"),
    #     Input("lng_input", "min"),
    #     Input("lat_input", "max"),
    #     Input("lng_input", "max"),
    # )
    # def invalid_button(lat, lng, lat_min, lng_min, lat_max, lng_max):
    #     # Devuelve True si la posición está fuera de los límites -> botón deshabilitado
    #     return (
    #         lat < float(lat_min) or lat > float(lat_max)
    #         or lng < float(lng_min) or lng > float(lng_max)
    #     )


    # -------------------------------------------------
    # Callback para descargar los datos como CSV
    # -------------------------------------------------
    # @APP.callback(
    #     Output("download-dataframe-csv", "data"),
    #     Input("btn_csv", "n_clicks"),  # Evento de click en el botón
    #     State("loc_marker", "position"),  # Posición del marcador en el mapa
    #     State("region", "value"),
    #     State("variable", "value"),
    #     State("start_month", "value"),
    #     State("end_month", "value"),
    #     prevent_initial_call=True,  # Evita que se ejecute al iniciar la app
    # )
    # def send_data_as_csv(
    #     n_clicks, marker_pos, region, variable, start_month, end_month,
    # ):
    #     # Extraer lat/lng del marcador
    #     lat = marker_pos[0]
    #     lng = marker_pos[1]

    #     # Convertir meses a enteros usando utilidades
    #     start_month = ac.strftimeb2int(start_month)
    #     end_month = ac.strftimeb2int(end_month)

    #     # Usar promedio multi-modelo
    #     model = "Multi-Model-Average"

    #     # Obtener datos locales filtrados
    #     data_ds, error_msg = local_data(
    #         lat, lng, region, model, variable, start_month, end_month
    #     )

    #     if error_msg == None :
    #         # Determinar hemisferio para unidades de lat/lng
    #         lng_units = "E" if (lng >= 0) else "W"
    #         lat_units = "N" if (lat >= 0) else "S"

    #         # Nombre del archivo CSV con información de fechas, variable y coordenadas
    #         file_name = (
    #             f'{data_ds["histo"]["T"].dt.strftime("%b")[0].values}-'
    #             f'{data_ds["histo"]["seasons_ends"].dt.strftime("%b")[0].values}'
    #             f'_{variable}_{abs(lat)}{lat_units}_{abs(lng)}{lng_units}'
    #             f'.csv'
    #         )

    #         # Convertir dataset a DataFrame y enviar como CSV
    #         df = data_ds.to_dataframe()
    #         return dash.dcc.send_data_frame(df.to_csv, file_name)
    #     else :
    #         # Retorna None si hay error en los datos
    #         return None

    # -------------------------------------------------
    # Callback para controlar panel de control
    # -------------------------------------------------
    @APP.callback(
    Output("period_container", "style"),
    Output("anomaly_period_container", "style"),
    Output("model_container", "style"),
    Output("period_years", "options"),           
    Output("period_years", "value"), 
    Input("data_type", "value")
    )
    def toggle_period_years(data_type):
        if data_type == "historical":
            period_style = {"display": "inline-block", "margin-left": "5px", "vertical-align": "top"}
            anomaly_period_style = {"display": "none"}
            model_style=anomaly_period_style 
            options = [
                {"label": "2006-2010", "value": "2006-2010"},
            ]
            value = "2006-2010"  # valor inicial por defecto
        elif data_type == "projected":
            period_style = {"display": "inline-block", "margin-left": "5px", "vertical-align": "top"}
            anomaly_period_style = {"display": "none"}
            model_style={"display": "inline-flex", "gap": "5px", "margin-left": "10px", "vertical-align": "top", "align-items": "flex-start"} 
            options = [
                
                    {"label": "2021-2025", "value": "2021-2025"},
                    {"label": "2026-2030", "value": "2026-2030"},
                    {"label": "2031-2035", "value": "2031-2035"},
                    {"label": "2036-2040", "value": "2036-2040"},
                    {"label": "2041-2045", "value": "2041-2045"},
                    {"label": "2046-2050", "value": "2046-2050"}
                 
            ]
            value = "2021-2025"
        else:  # anomaly, direction_change
            period_style = {"display": "none"}
            anomaly_period_style = {"display": "inline-block", "margin-left": "20px", "vertical-align": "top"}
            model_style=anomaly_period_style 
            options = []
            value = None

        return period_style, anomaly_period_style, model_style,options, value
    # -------------------------------------------------
    # Callback para generar el gráfico local de serie de tiempo
    # -------------------------------------------------
    @APP.callback(
        Output("local_graph", "figure"),
        Input("geojson", "clickData"),
        Input("geojson", "dblclickData"),
        Input("geojson", "click_feature"),
        #Input("edit-control", "geojson"),
        Input("graph_type", "value"),
        Input("variety", "options"),
        Input("hist_years", "options"),
        Input("fcst_years", "options"),        
        )
    def local_plots( click,dblclick,mouseover,type,variety,hist_years,fcst_years) :
        local_graph = None
        fig=None
        ctx = dash.callback_context
        print(mouseover)
        event = ctx.triggered[0]["prop_id"]
        variety_values = [opt["value"] for opt in variety]
        hist_years_values = [opt["value"] for opt in hist_years]
        fcst_years_values = [opt["value"] for opt in fcst_years]
        #print(click["properties"]["id"])
        #pepe=extraf.cargar_valores_id("data/pepsi", click["properties"]["id"])
        #print(data)
        print(event)
        if click is not None and click.get("properties") is not None:
            data=[extrafunctions.cargar_valores_id("data/cvs_files", 
                                                   click["properties"]["id"],
                                                   "HARWT",
                                                   variety_values,
                                                   hist_years_values,
                                                   fcst_years_values 
                                                   )
                                                   ]
            #print(data)
            fig = pgo.Figure()

            #type = "bar" 
            # Agregar cada trace a la figura
            # Escalar automáticamente los tamaños según Y
            min_size = 10
            max_size = 50

            #y_array = np.array([])
            #sizes = min_size + (y_array - y_array.min()) / (y_array.max() - y_array.min()) * (max_size - min_size)

            for trace in data:
                if type == "bars":
                    fig.add_trace(pgo.Bar(x=trace["x"], y=trace["y"], name=click["properties"]["NAME"]))
                elif type in ["lines","markers","lines+markers"]:
                    fig.add_trace(pgo.Scatter(x=trace["x"], y=trace["y"],mode=type, name=click["properties"]["NAME"]))
                elif type in ["markers-scale"]:
                    #y_array = np.append(trace["y"])
                    #print(trace["y"])
                    y_array = np.array(trace["y"])
                    fig.add_trace(pgo.Scatter(x=trace["x"], y=trace["y"],mode="markers", name=click["properties"]["NAME"],
                                              marker=dict(
                                                    size=trace["y"],         # tamaño variable
                                                    #color="blue",       # color de los círculos
                                                    sizemode="area",    # el tamaño corresponde al área
                                                    sizeref=2.*max(trace["y"])/(40.**2),  # escala del tamaño (opcional)
                                                    #sizeref=0.1 * 2.*max(trace["y"])/(40.**2),
                                                    sizemin=4,           # tamaño mínimo
                                                    color=trace["y"],   # color según valor
                                                    colorscale=layout.colorscale,
                                                    showscale=True
                                                ) 
                                              ))
                
            fig.update_layout(
                title=f"{click['properties']['NAME']} , {click['properties']['STATE_NAME']}",
                xaxis_title="Variety",
                yaxis_title="HARWT",
            )
            # USAR PARA AGRUPAR VARIOS Tipos de graficos
            #fig.update_layout(barmode="group")

            # fig = {
            # "data": [data],
            # "layout": {
            #     "title": f"Valores encontrados para ID {click['properties']['id']}",
            #     "xaxis": {"title": "Archivo"},
            #     "yaxis": {"title": "Valor"},
            #     }
            # }
        return fig
        #return local_graph
    # @APP.callback(
    #     Output("local_graph", "figure"),  # Actualiza la figura del gráfico
    #     Input("loc_marker", "position"),  # Posición seleccionada en el mapa
    #     Input("region", "value"),  # Región seleccionada
    #     Input("submit_controls","n_clicks"),  # Botón para actualizar el gráfico
    #     State("model", "value"),  # Modelo seleccionado
    #     State("variable", "value"),  # Variable seleccionada
    #     State("start_month", "value"),  # Mes de inicio del periodo
    #     State("end_month", "value"),  # Mes de fin del periodo
    #     State("start_year", "value"),  # Año de inicio del periodo proyectado
    #     State("end_year", "value"),  # Año de fin del periodo proyectado
    #     State("start_year_ref", "value"),  # Año de inicio del periodo de referencia
    #     State("end_year_ref", "value"),  # Año de fin del periodo de referencia
    # )
    # def local_plots(
    #     marker_pos,
    #     region,
    #     n_clicks,
    #     model,
    #     variable,
    #     start_month,
    #     end_month,
    #     start_year,
    #     end_year,
    #     start_year_ref,
    #     end_year_ref,
    # ):
    #     # Extraer latitud y longitud del marcador
    #     lat = marker_pos[0]
    #     lng = marker_pos[1]

    #     # Convertir meses de string a entero
    #     start_month = ac.strftimeb2int(start_month)
    #     end_month = ac.strftimeb2int(end_month)

    #     # Obtener los datos locales filtrados por lat/lng, región, modelo, variable y periodo
    #     data_ds, error_msg = local_data(
    #         lat, lng, region, model, variable, start_month, end_month
    #     )

    #     if error_msg != None :
    #         # Si hay error, se genera un gráfico de error
    #         local_graph = pingrid.error_fig(error_msg)
    #     else :
    #         # Determinar formato de fecha en el eje X según si el periodo cruza años
    #         if (end_month < start_month) :
    #             start_format = "%b %Y - "
    #         else:
    #             start_format = "%b-"

    #         # Inicializar figura vacía
    #         local_graph = pgo.Figure()

    #         # Diccionario de colores por variable/modelo
    #         data_color = {
    #             "histo": "blue", "picontrol": "green",
    #             "ssp126": "yellow", "ssp370": "orange", "ssp585": "red",
    #         }

    #         # Unidades geográficas para el título
    #         lng_units = "˚E" if (lng >= 0) else "˚W"
    #         lat_units = "˚N" if (lat >= 0) else "˚S"

    #         # Agregar una serie de tiempo por cada variable del dataset
    #         for var in data_ds.data_vars:
    #             local_graph.add_trace(plot_ts(
    #                 data_ds[var].mean("M", keep_attrs=True),  # Promedio sobre modelos si es Multi-Model-Average
    #                 var,
    #                 data_color[var],
    #                 start_format,
    #                 data_ds[var].attrs["units"]  # Unidades de la variable
    #             ))

    #         # Agregar rectángulo para periodo de referencia
    #         add_period_shape(
    #             local_graph,
    #             data_ds,
    #             start_year_ref,
    #             end_year_ref,
    #             "blue",
    #             "RoyalBlue",
    #             "reference period",
    #         )

    #         # Agregar rectángulo para periodo proyectado
    #         add_period_shape(
    #             local_graph,
    #             data_ds,
    #             start_year,
    #             end_year,
    #             "LightPink",
    #             "Crimson",
    #             "projected period",
    #         )

    #         # Actualizar layout del gráfico: títulos, márgenes y etiquetas de ejes
    #         local_graph.update_layout(
    #             xaxis_title="Time",
    #             yaxis_title=(
    #                 f'{data_ds["histo"].attrs["long_name"]} '
    #                 f'({data_ds["histo"].attrs["units"]})'
    #             ),
    #             title={
    #                 "text": (
    #                     f'{data_ds["histo"]["T"].dt.strftime("%b")[0].values}-'
    #                     f'{data_ds["histo"]["seasons_ends"].dt.strftime("%b")[0].values}'
    #                     f' {variable} seasonal average from model {model} '
    #                     f'at ({abs(lat)}{lat_units}, {abs(lng)}{lng_units})'
    #                 ),
    #                 "font": dict(size=14),
    #             },
    #             margin=dict(l=30, r=30, t=30, b=30),
    #         )

    #     return local_graph


    # -------------------------------------------------
    # Callback para generar descripción textual del mapa
    # -------------------------------------------------
    # @APP.callback(
    #     Output("map_description", "children"),  # Actualiza el texto descriptivo del mapa
    #     Input("submit_controls", "n_clicks"),  # Botón para actualizar la descripción
    #     State("scenario", "value"),  # Escenario seleccionado
    #     State("model", "value"),  # Modelo seleccionado
    #     State("variable", "value"),  # Variable seleccionada
    #     State("start_month", "value"),  # Mes de inicio del periodo proyectado
    #     State("end_month", "value"),  # Mes de fin del periodo proyectado
    #     State("start_year", "value"),  # Año de inicio del periodo proyectado
    #     State("end_year", "value"),  # Año de fin del periodo proyectado
    #     State("start_year_ref", "value"),  # Año de inicio del periodo de referencia
    #     State("end_year_ref", "value"),  # Año de fin del periodo de referencia
    # )
    # def write_map_description(
    #     n_clicks,
    #     scenario,
    #     model,
    #     variable,
    #     start_month,
    #     end_month,
    #     start_year,
    #     end_year,
    #     start_year_ref,
    #     end_year_ref,
    # ):
    #     # Construye la descripción textual del mapa usando los parámetros seleccionados
    #     return (
    #         f'The Map displays the change in {start_month}-{end_month} seasonal average of '
    #         f'{variable} from {model} model under {scenario} scenario projected for '
    #         f'{start_year}-{end_year} with respect to historical {start_year_ref}-'
    #         f'{end_year_ref}'
    #     )

    # -------------------------------------------------
    # Callback para generar descripcion 
    # -------------------------------------------------
    @APP.callback(
        Output("map_description_var", "children"),
        Input("data_type", "value"),
        State("data_type", "options")
    )
    def update_map_description(variable,options):
        label = next(
        (opt["label"] for opt in options if opt["value"] == variable),
        variable
        )
        description = [
        html.B("Description: "),
        html.Span(f"Projected changes for variable '{label}'."),
        html.Br(),
        html.Small("Based on selected model and reference period. ")
        ]

        if variable == "mean_change":
            description.extend(
                [
                html.Small("Mean change shows how the future average differs from the historical average. Positive " \
                            "values indicate an increase; negative values indicate a decrease relative " \
                            "to the historical period."),
                html.Br(),
                html.B("Equation used: "),
                html.Small([
                    "mean",
                    html.Sub("fcst"),
                    " − mean",
                    html.Sub("hist"),
                ])
                ]
            )
        elif variable == "percentage_change":
            description.extend(
                [
                html.Small(
                    "Percentage change shows the relative difference between the projected and historical averages. "\
                    "Positive values indicate an increase; negative values indicate a decrease relative to the historical period."
                ),
                html.Br(),
                html.B("Equation used: "),
                html.Small([
                        "(",
                        "mean",
                        html.Sub("fcst"),
                        " − ",
                        "mean",
                        html.Sub("hist"),
                        ") / ",
                        "mean",
                        html.Sub("hist"),
                        " × 100",
                    ])
                ]
            )
        elif variable == "direction_change":
            description.extend(
                [
                html.Small(
                    "Direction change indicates whether the variable increases, decreases, or stays the same. "\
                    "A value of +1 indicates an increase, 0 indicates no change, and -1 indicates a decrease relative to the historical period."
                ),
                html.Br(),
                html.B("Equation used: "),
                html.Small([
                        "sign(",
                        "mean",
                        html.Sub("fcst"),
                        " − ",
                        "mean",
                        html.Sub("hist"),
                        ")"
                    ])
                ]
            )
        elif variable == "stress_simple":
            description.extend(
                [
                html.Small(
                    "Stress index shows how much the future deviates from historical conditions. "\
                    "Values near -1 indicate extreme stress, 0 indicates neutral conditions, "\
                    "and values near 1 indicate extreme benefit relative to the historical period."
                ),
                html.Br(),
                html.B("Equation used: "),
                html.Small([
                        "1 − (",
                        "mean",
                        html.Sub("fcst"),
                        " / ",
                        "mean",
                        html.Sub("hist"),
                        ")"
                    ])
                ]
            )
        elif variable == "yield_change_index":
            description.extend(
                [
                html.Small(
                    "Yield Change Index (YCI%) shows the relative change in yield compared to historical conditions. "\
                    "Positive values indicate increased yield, negative values indicate decreased yield relative to the historical period."
                ),
                html.Br(),
                html.B("Equation used: "),
                html.Small([
                    "(",
                    "mean",
                    html.Sub("fcst"),
                    " / ",
                    "mean",
                    html.Sub("hist"),
                    " − 1) × 100"
                ])
                ]
            )

        return description
        # -------------------------------------------------
    # Callback para generar el título del mapa
    # -------------------------------------------------
    @APP.callback(
        Output("map_title", "children"),  # Actualiza el texto del título del mapa
        #Input("submit_controls","n_clicks"),  # Evento al hacer click en el botón de actualizar
        #State("scenario", "value"),  # Escenario seleccionado
        Input("model", "value"),  # Modelo seleccionado
        Input("variable", "value"),  # Variable seleccionada
        Input("variety", "value"),
        Input("period_years", "value"),
        Input("data_type", "value"),
        Input("anomaly_period_values", "data")
        #State("start_month", "value"),  # Mes de inicio del periodo proyectado
        #State("end_month", "value"),  # Mes de fin del periodo proyectado
        #State("start_year", "value"),  # Año de inicio del periodo proyectado
        #State("end_year", "value"),  # Año de fin del periodo proyectado
        #State("start_year_ref", "value"),  # Año de inicio del periodo de referencia
        #State("end_year_ref", "value"),  # Año de fin del periodo de referencia
    )
    def write_map_title(
        #n_clicks,
        #scenario,
        model,
        variable,
        variety,
        period_years,
        data_type,
        anom_period
        #start_month,
        #end_month,
        #start_year,
        #end_year,
        #start_year_ref,
        #end_year_ref,
    ):
        if data_type not in ['historical','projected']:
            top_title=f"{model} {variable} {data_type.replace('_', ' ').capitalize()} {variety} for {'/'.join(anom_period)}"
        else:
            top_title=f"{model} {variable} change from {variety} for {period_years}"
        # Construye un título resumido del mapa mostrando periodo, escenario, modelo, variable y referencia
        return (top_title)

    # -----------------------------
    # Callback para cargar CSV según dropdown
    # -----------------------------
    # Callback para combinar valores
    @APP.callback(
        Output("anomaly_period_values", "data"),
        [Input("hist_years", "value"), Input("fcst_years", "value")]
    )
    def combine_inputs(hist, fcst):
        return [hist, fcst]
    
    @APP.callback(
        Output("geojson", "data"),
        Output("geojson", "hideout"),
        Output("colorbar", "children"),
        Input("variety", "value"),
        Input("model", "value"),
        Input("scenario", "value"),
        Input("period_years", "value"),
        Input("data_type", "value"),
        Input("anomaly_period_values", "data")
        #[Input("hist_years", "value"), Input("fcst_years", "value")]
    )
    def load_csv_and_update(variety,model,scenario,target_value,data_type,anom_period):
        # #print("Pepe")
        
        #print(data_type)
        #print(anom_period[0])
        # #print(f"layout es {layout.data}")
        # #print("Callback load_csv_and_update disparado:", variety)
        
        # # Seleccionar CSV
        # csv_file = f"data/pepsi/RC4BL_HARWT_ID_PDhist_{variety}_{target_value}.csv"
        # # if variety == "1867_2021-2025":
        # #     csv_file = "data/pepsi/RC4BL_HARWT_ID_PDhist_1867_2021-2025.csv"
        # # else:
        # #     csv_file = "data/pepsi/RC4BL_HARWT_ID_PDhist_1867_2046-2050.csv"

        # df = pd.read_csv(csv_file)
        # if "CCSUID" in df.columns:
        #     df["id"] = df["CCSUID"].astype(str)
        # elif "FIPS" in df.columns:
        #     df["id"] = df["FIPS"].astype(str)
        # df["HARWT"] = pd.to_numeric(df["HARWT"], errors="coerce")
        # var_map = df.set_index("id")["HARWT"].to_dict()

        # # Actualizar geojson (solo data, no reemplazar GeoJSON completo)
        # new_features = []
        # for feature in layout.data["features"]:
        #     fid = str(feature["properties"].get("id", ""))
        #     feature["properties"]["HARWT"] = var_map.get(fid, np.nan)
        #     new_features.append(feature)

        # data_filtered = {
        #     "type": "FeatureCollection",
        #     "features": [f for f in new_features if f["properties"].get("HARWT") is not None and not np.isnan(f["properties"].get("HARWT"))]
        # }

        # # Calcular nuevas clases
        # min_val = df["HARWT"].min()
        # max_val = df["HARWT"].max()

        data_filtered,new_classes,colorscale=extrafunctions.prepare_data(variety,model,scenario,target_value,data_type,anom_period)
        #base = layout.calcular_base(min_val, max_val)
        #new_classes = layout.generar_clases(min_val, max_val, base=base)
        #print(new_classes)
        hideout = dict(colorscale=colorscale, classes=new_classes, style=layout.style_default, colorProp="HARWT")
        #print(hideout)
        # Colorbar
        
        if data_type not in ['historical','projected']:
            title=f"HARWT {data_type.replace('_', ' ').capitalize()} variety={variety} {'/'.join(anom_period)}"
        else:
            title=f"HARWT variety={variety} target={target_value}"

        if data_type in ['percentage_change','yield_change_index']:
            ctg = ["{}%".format(cls) for cls in new_classes]
        elif data_type == 'direction_change':
            ctg = ["{}".format(cls) for cls in ['Decrease','No change','Increase']]
        else:
            ctg = ["{}".format(cls) for cls in new_classes[:-1]] + ["{}+".format(new_classes[-1])]
        new_colorbar = html.Div(
            children=[
                html.Div(style={
                    "position": "absolute", "bottom": "0", "left": "0",
                    "width": "400px", "height": "20px", "background": "rgba(255,255,255,0.9)",
                    "zIndex": 0
                }),
                html.Div([
                    html.Div(title, style={"fontWeight": "bold", "marginBottom": "5px", "zIndex": 1}),
                    dlx.categorical_colorbar(categories=ctg, colorscale=colorscale, width=400, height=30, position="bottomleft")
                ], style={"position": "relative", "zIndex": 1})
            ],
            style={"position": "absolute", "bottom": "10px", "left": "10px", "width": "420px", "height": "70px", "zIndex": 999}
        )

        # IMPORTANTE: no cambiamos 'key', así el GeoJSON no se destruye
        #print(data_filtered)
        return data_filtered, hideout, new_colorbar
    
    @APP.callback(
        Output("overlay-borders", "checked"),
        Input("geojson", "data"),  # <- se dispara cuando el GeoJSON ya está listo
        prevent_initial_call=True
    )
    def toggle_borders_after_update(_):
        # 1er paso: apagar bordes
        return False

    @APP.callback(
        Output("overlay-borders", "checked", allow_duplicate=True),
        Input("overlay-borders", "checked"),
        prevent_initial_call=True
    )
    def turn_borders_on(checked):
        # Cuando se apague, lo prendemos
        if checked is False:
            return True
        return dash.no_update
    
    # -----------------------------
    # Callback hover/click info
    # -----------------------------
    @APP.callback(
        Output("info", "children"),
        #[
        Input("geojson", "hoverData"),
        #Input("geojson", "clickData")
        #]
    )
    def update_info(hover):#, click):
        #print(f"el click es: {click}")
        #import dash
        ctx = dash.callback_context
        if not ctx.triggered:
            return layout.get_info()
        prop_id = ctx.triggered[0]["prop_id"]
        #print(prop_id)
        if prop_id == "geojson.hoverData":
            feature = hover
        # elif prop_id == "geojson.click_feature":
        #     #print("es un clic")
        #     feature = click
        else:
            feature = None
        return layout.get_info(feature)
    

    # -------------------------------------------------
    # Validación de años usando clientside callback
    # -------------------------------------------------
    # APP.clientside_callback(
    #     """function(start_year, end_year, start_year_ref, end_year_ref) {
    #         // Validar periodo proyectado
    #         if (start_year && end_year) {
    #             invalid_start_year = (start_year > end_year)
    #             invalid_end_year = invalid_start_year
    #         } else {
    #             invalid_start_year = !start_year
    #             invalid_end_year = !end_year
    #         }

    #         // Validar periodo de referencia
    #         if (start_year_ref && end_year_ref) {
    #             invalid_start_year_ref = (start_year_ref > end_year_ref)
    #             invalid_end_year_ref = invalid_start_year_ref
    #         } else {
    #             invalid_start_year_ref = !start_year_ref
    #             invalid_end_year_ref = !end_year_ref
    #         }

    #         // Retorna los flags de validación para cada input y si el botón debe deshabilitarse
    #         return [
    #             invalid_start_year, invalid_end_year,
    #             invalid_start_year_ref, invalid_end_year_ref,
    #             (
    #                 invalid_start_year || invalid_end_year
    #                 || invalid_start_year_ref || invalid_end_year_ref
    #             ),
    #         ]
    #     }
    #     """,
    #     Output("start_year", "invalid"),
    #     Output("end_year", "invalid"),
    #     Output("start_year_ref", "invalid"),
    #     Output("end_year_ref", "invalid"),
    #     Output("submit_controls", "disabled"),
    #     Input("start_year", "value"),
    #     Input("end_year", "value"),
    #     Input("start_year_ref", "value"),
    #     Input("end_year_ref", "value"),
    # )


    # # -------------------------------------------------
    # # Función para calcular el cambio estacional relativo a un periodo de referencia
    # # -------------------------------------------------
    # def seasonal_change(
    #     scenario,
    #     model,
    #     variable,
    #     region,
    #     start_month,
    #     end_month,
    #     start_year,
    #     end_year,
    #     start_year_ref,
    #     end_year_ref,
    # ):
    #     # Expandir Multi-Model-Average a lista de modelos individuales
    #     model = [model] if model != "Multi-Model-Average" else [
    #         "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR","MRI-ESM2-0", "UKESM1-0-LL"
    #     ]

    #     # Calcular periodo de referencia histórico
    #     ref = xr.concat([
    #         ac.seasonal_data(
    #             ac.read_data("historical", m, variable, region, unit_convert=True),
    #             start_month, end_month,
    #             start_year=start_year_ref, end_year=end_year_ref,
    #         ).mean(dim="T", keep_attrs=True) for m in model
    #     ], "M").mean("M", keep_attrs=True)

    #     # Calcular periodo proyectado
    #     data = xr.concat([
    #         ac.seasonal_data(
    #             ac.read_data(scenario, m, variable, region, unit_convert=True),
    #             start_month, end_month,
    #             start_year=start_year, end_year=end_year,
    #         ).mean(dim="T", keep_attrs=True) for m in model
    #     ], "M").mean("M", keep_attrs=True)

    #     # Restar referencia para obtener cambio manteniendo atributos
    #     data = xr.apply_ufunc(
    #         np.subtract, data, ref, dask="allowed", keep_attrs="drop_conflicts",
    #     )

    #     # Convertir a porcentaje si corresponde
    #     if variable in ["hurs", "huss", "pr"]:
    #         data = 100. * data / ref
    #         data.attrs["units"] = "%"

    #     # Renombrar coordenadas para compatibilidad con mapas
    #     return data.rename({"X": "lon", "Y": "lat"})


    # # -------------------------------------------------
    # # Función para determinar atributos de visualización del mapa (colormap, min, max)
    # # -------------------------------------------------
    # def map_attributes(data):
    #     variable = data.name

    #     if variable in ["tas", "tasmin", "tasmax"]:
    #         colorscale = CMAPS["temp_anomaly"]
    #     elif variable in ["hurs", "huss"]:
    #         colorscale = CMAPS["prcp_anomaly"].rescaled(-30, 30)
    #     elif variable in ["pr"]:
    #         colorscale = CMAPS["prcp_anomaly"].rescaled(-100, 100)
    #     else:
    #         # Calcular amplitud máxima para normalizar colormap
    #         map_amp = np.max(np.abs(data)).values
    #         if variable in ["prsn"]:
    #             colorscale = CMAPS["prcp_anomaly_blue"]
    #         elif variable in ["sfcwind"]:
    #             colorscale = CMAPS["std_anomaly"]
    #         else:
    #             colorscale = CMAPS["correlation"]
    #         colorscale = colorscale.rescaled(-1*map_amp, map_amp)

    #     return colorscale, colorscale.scale[0], colorscale.scale[-1]

    # -------------------------------------------------
    # Callback para dibujar la barra de colores del mapa
    # -------------------------------------------------
    # @APP.callback(
    #     Output("colorbar", "colorscale"),
    #     Output("colorbar", "min"),
    #     Output("colorbar", "max"),
    #     Output("colorbar", "unit"),
    #     Input("region", "value"),
    #     Input("submit_controls","n_clicks"),
    #     State("scenario", "value"),
    #     State("model", "value"),
    #     State("variable", "value"),
    #     State("start_month", "value"),
    #     State("end_month", "value"),
    #     State("start_year", "value"),
    #     State("end_year", "value"),
    #     State("start_year_ref", "value"),
    #     State("end_year_ref", "value"),
    # )
    # def draw_colorbar(
    #     region,
    #     n_clicks,
    #     scenario,
    #     model,
    #     variable,
    #     start_month,
    #     end_month,
    #     start_year,
    #     end_year,
    #     start_year_ref,
    #     end_year_ref,
    # ):
    #     # Expandir Multi-Model-Average a lista de modelos
    #     model = [model] if model != "Multi-Model-Average" else [
    #         "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR","MRI-ESM2-0", "UKESM1-0-LL"
    #     ]

    #     # Calcular cambio estacional promedio sobre todos los modelos
    #     data = xr.concat([seasonal_change(
    #         scenario,
    #         m,
    #         variable,
    #         region,
    #         ac.strftimeb2int(start_month),
    #         ac.strftimeb2int(end_month),
    #         int(start_year),
    #         int(end_year),
    #         int(start_year_ref),
    #         int(end_year_ref),
    #     ) for m in model], "M").mean("M", keep_attrs=True)

    #     # Obtener colormap, min y max
    #     colorbar, min, max = map_attributes(data)
    #     return colorbar.to_dash_leaflet(), min, max, data.attrs["units"]


    # -------------------------------------------------
    # Callback para crear los layers del mapa y advertencias
    # -------------------------------------------------
    # @APP.callback(
    #     Output("layers_control", "children"),
    #     Output("map_warning", "is_open"),
    #     Input("region", "value"),
    #     Input("submit_controls", "n_clicks"),
    #     State("scenario", "value"),
    #     State("model", "value"),
    #     State("variable", "value"),
    #     State("start_month", "value"),
    #     State("end_month", "value"),
    #     State("start_year", "value"),
    #     State("end_year", "value"),
    #     State("start_year_ref", "value"),
    #     State("end_year_ref", "value"),
    # )
    # def make_map(
    #     region,
    #     n_clicks,
    #     scenario,
    #     model,
    #     variable,
    #     start_month,
    #     end_month,
    #     start_year,
    #     end_year,
    #     start_year_ref,
    #     end_year_ref,
    # ):
    #     try:
    #         send_alarm = False
    #         # Construir URL de los tiles según parámetros seleccionados
    #         url_str = (
    #             f"{TILE_PFX}/{{z}}/{{x}}/{{y}}/{region}/{scenario}/{model}/{variable}/"
    #             f"{start_month}/{end_month}/{start_year}/{end_year}/{start_year_ref}/"
    #             f"{end_year_ref}"
    #         )
    #     except:
    #         # En caso de error, activar alarma y dejar URL vacía
    #         url_str= ""
    #         send_alarm = True

    #     # Generar capas de mapa y devolver estado de advertencia
    #     return mru.layers_controls(
    #         url_str, f"change_{region}", "Change",
    #         GLOBAL_CONFIG["datasets"][f"shapes_adm_{region}"], GLOBAL_CONFIG,
    #         adm_id_suffix=region,
    #     ), send_alarm


    # -------------------------------------------------
    # Ruta Flask para servir los tiles del mapa
    # -------------------------------------------------
    # @FLASK.route(
    #     (
    #         f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>/<region>/<scenario>/<model>/<variable>/"
    #         f"<start_month>/<end_month>/<start_year>/<end_year>/<start_year_ref>/"
    #         f"<end_year_ref>"
    #     ),
    #     endpoint=f"{config['core_path']}"
    # )
    # def fcst_tiles(tz, tx, ty,
    #     region,
    #     scenario,
    #     model,
    #     variable,
    #     start_month,
    #     end_month,
    #     start_year,
    #     end_year,
    #     start_year_ref,
    #     end_year_ref,
    # ):
    #     # Expandir Multi-Model-Average a lista de modelos
    #     model = [model] if model != "Multi-Model-Average" else [
    #         "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR","MRI-ESM2-0", "UKESM1-0-LL"
    #     ]

    #     # Calcular datos de cambio estacional promedio
    #     data = xr.concat([seasonal_change(
    #         scenario,
    #         m,
    #         variable,
    #         region,
    #         ac.strftimeb2int(start_month),
    #         ac.strftimeb2int(end_month),
    #         int(start_year),
    #         int(end_year),
    #         int(start_year_ref),
    #         int(end_year_ref),
    #     ) for m in model], "M").mean("M", keep_attrs=True)

    #     # Asignar atributos de colormap y escala
    #     (
    #         data.attrs["colormap"],
    #         data.attrs["scale_min"],
    #         data.attrs["scale_max"],
    #     ) = map_attributes(data)

    #     # Generar y devolver tile del mapa
    #     resp = pingrid.tile(data, tx, ty, tz)
    #     return resp
