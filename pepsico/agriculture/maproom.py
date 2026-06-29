import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
from . import layout  
from . import extrafunctions  
import plotly.graph_objects as pgo
from globals_ import FLASK, GLOBAL_CONFIG  # Global configuration settings
import numpy as np
from dash_extensions.enrich import html
import dash_leaflet.express as dlx
import dash_auth
import pandas as pd


HISTORICAL_VALUES = tuple(m["value"] for m in layout.HISTORICAL_YEARS)
PROJECTED_VALUES = tuple(m["value"] for m in layout.PROJECTED_YEARS)

def register(FLASK, config):
    # -------------------------------------------------
    # URL prefixes for the app and tile endpoints
    # -------------------------------------------------
    PFX = f"{GLOBAL_CONFIG['url_path_prefix']}/{config['core_path']}"
    TILE_PFX = f"{PFX}/tile"

    # -------------------------------------------------
    # Dash app initialization
    # -------------------------------------------------
    APP = dash.Dash(
        __name__,
        server=FLASK,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,  # Bootstrap theme for the UI
        ],
        url_base_pathname=f"{PFX}/",  # Base URL de la app
        meta_tags=[  # Meta tags para responsividad 
            {"name": "description", "content": "Forecast"},
            {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
        ],
    )
    
    # -------------------------------------------------
    # Authentication setup
    # Loads hashed user credentials and validates them
    # using bcrypt before granting access to the app.
    # -------------------------------------------------
    VALID_USERS = extrafunctions.load_valid_users_hash()
    def bcrypt_auth(username, password):
        if username not in VALID_USERS:
            return False
        return extrafunctions.verify_password(password, VALID_USERS[username])
    dash_auth.BasicAuth(APP, auth_func=bcrypt_auth,secret_key="aXJpZGw=")


    APP.enable_dev_tools(debug=True)
    
    APP.title = "Forecast"  
    # App layout definition
    APP.layout = layout.app_layout()

    # -------------------------------------------------
    # Callback: Download graph data as CSV
    # Triggered by the download button; reads the
    # current graph data store and county info to
    # build a long-format DataFrame (Model, Period,
    # Value) and sends it as a CSV file download.
    # -------------------------------------------------
    @APP.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    State("graph_data_store", "data"),
    State("graph_county_info", "data"),
    prevent_initial_call=True,
    )
    def download_csv(n_clicks, graph_data,county_info):
        graph_data=graph_data[0]
        periods = graph_data["x_axis"]

        rows = []

        for model, values in graph_data["models"].items():

            for period, value in zip(periods, values):

                rows.append({
                    "Model": model,
                    "Period": period,
                    "Value": value
                })

        df = pd.DataFrame(rows)

        return dash.dcc.send_data_frame(
            df.to_csv,
            f"graph_data_{county_info}.csv",
            index=False,
        )

    # -------------------------------------------------
    # Callback: Control panel visibility toggling
    # Adjusts the visibility of dataset containers,
    # planting/scenario selectors, and year dropdowns
    # depending on whether the secondary dataset type
    # is set to "historical" or a projected scenario.
    # -------------------------------------------------
    @APP.callback(
    Output("dataset2_container", "style"),
    Output("dataset1_container", "style"),
    Output("dataset2_plant-scen_container", "style"),
    Output("dataset2_years", "options"),
    Output("dataset2_years", "value"),
    Output("dataset1", "options"),
    Output("dataset1_years", "options"),
    Input("dataset2", "value"),
    Input("dataset1", "value"),
     )
    def toggle_multidataset(dataset2, dataset1):
        dataset2_years_options =[]
        dataset1_years_options =[]
        dataset2_planting_style={"display": "none"} 
        dataset2_scenario_style=dataset2_planting_style
        if  dataset2 == "historical" : #and data_type not in  ["historical", "projected"]) 
            dataset2_years_options = layout.HISTORICAL_YEARS
            dataset1 = layout.MODELS
            dataset1_years_options = layout.PROJECTED_YEARS
            dataset2_style= {"display": "flex", "flex-direction": "column"}
            dataset1_style=dataset2_style
            dataset2_planting_style={"display": "none"} 
            dataset2_scenario_style=dataset2_planting_style
            dataset2_value = layout.HISTORICAL_YEARS[0]['value'] 
        else: 
            dataset2_years_options = layout.PROJECTED_YEARS
            dataset1 = layout.MODELS
            dataset1_years_options = layout.PROJECTED_YEARS
            dataset2_style={"display": "flex", "flex-direction": "column"}
            dataset1_style=dataset2_style
            dataset2_planting_style={"display": "flex", "flex-direction": "column"}
            dataset2_scenario_style=dataset2_style
            dataset2_value = layout.PROJECTED_YEARS[0]['value'] 

        return dataset2_style,dataset1_style,dataset2_scenario_style, dataset2_years_options,dataset2_value,dataset1,dataset1_years_options

    # -------------------------------------------------
    # Callback: Period/year controls and info panel
    # Shows or hides the period selector, anomaly
    # period selector, and model selector based on the
    # selected data type (historical, projected, or an
    # anomaly/direction-change view). Also adjusts the
    # vertical position of the info panel accordingly.
    # -------------------------------------------------
    @APP.callback(
    Output("period_container", "style"),
    Output("anomaly_period_container", "style"),
    Output("model_container", "style"),
    Output("info", "style"),
    Output("period_years", "options"),           
    Output("period_years", "value"), 
    Input("data_type", "value"),
    Input("info", "style"),
    )
    def toggle_period_years(data_type,info_style):
        options=[]
        info_style['top'] = '200px'
        if data_type == "historical" :
            period_style = {"display": "inline-block", "margin-left": "5px", "vertical-align": "top"}
            anomaly_period_style = {"display": "none"}
            model_style=anomaly_period_style
            options = layout.HISTORICAL_YEARS
            value = layout.HISTORICAL_YEARS[0]['value']  # valor inicial por defecto
            #"top": "200px"
        elif data_type == "projected":
            period_style = {"display": "inline-block", "margin-left": "5px", "vertical-align": "top"}
            anomaly_period_style = {"display": "none"}
            model_style={"display": "inline-flex", "gap": "5px", "margin-left": "10px", "vertical-align": "top", "align-items": "flex-start"} 
            options = layout.PROJECTED_YEARS
            value = layout.PROJECTED_YEARS[0]['value']
        else:  # anomaly, direction_change
            period_style = {"display": "none"}
            anomaly_period_style = {"display": "inline-block", "margin-left": "20px", "vertical-align": "top"}
            model_style=period_style
            value = None
            info_style['top'] = '310px'

        return period_style, anomaly_period_style, model_style,info_style,options, value, 
    
    # -------------------------------------------------
    # Callback: Show/hide scenario selector for bar charts
    # The graph scenario container is only relevant
    # when the selected graph type is a bar chart
    # (type prefix "bars"). For line/marker charts it
    # is hidden.
    # -------------------------------------------------
    @APP.callback(
        Output("graph_scenario_container", "style"),
        Input("graph_type","value")
    )
    def local_plots_display(graph_type):
        if graph_type.split("_")[0] == "bars":
            return {"display": "flex", "align-items": "center"}
        else:
            return {"display": "none"}
    # -------------------------------------------------
    # Shared input map for the local graph callback
    # Groups variety, variable, planting style, graph
    # type, and scenario inputs for reuse.
    # -------------------------------------------------
    options = { 'variety':Input("variety", "value"), #args[0]
                 'variable':Input("variable", "options"), #args[0]
                 'planting':Input("graph_planting", "value"),
                 'graph_type':Input("graph_type", "value"),
                 'graph_scenario':Input("graph_scenario", "value"),
                      }

    # -------------------------------------------------
    # Callback: Local county graph rendering
    # Fires on map click. Loads bar or line/marker data
    # for the clicked county and builds a Plotly figure.
    # Handles three chart types:
    #   - "bars_*"       → grouped/stacked bar chart
    #   - "lines" / "markers" / "lines+markers" → scatter
    #   - "markers-scale" → bubble chart
    # A light-blue background marks the historical
    # period; light-red marks projected periods.
    # Historical data is always rendered as a reference
    # line/marker on top of projected model traces.
    # -------------------------------------------------
    @APP.callback(
        Output("local_graph", "figure"),
        Output("graph_data_store", "data"),
        Output("graph_county_info", "data"),
        Input("geojson", "clickData"),
        Input("geojson", "dblclickData"),
        Input("geojson", "click_feature"),
        Input("variety", "options"),
        Input("dataset1_years", "options"),
        Input("dataset2_years", "options"),
        options,
        prevent_initial_call=True,        
        )
    def local_plots( click,dblclick,mouseover,variety,dataset1_years,dataset2_years,*args) :

        def duplicate_points(x_axis, values):
            """Duplica cada punto con un pequeño offset en x para forzar una línea corta."""
            x_dup = []
            y_dup = []
            for i, (x, y) in enumerate(zip(x_axis, values)):
                x_dup.extend([i - 0.1, i + 0.1])  # ±0.1 offset around the index
                y_dup.extend([y, y])
            return x_dup, y_dup

        fig=None
        data=None
        county_info=None
        ctx = dash.callback_context
        variety_values = args[0]['variety'] 
        planting_values = args[0]['planting'] 
        type=args[0]['graph_type'] 
        
        if type.split("_")[0] == "bars":
            scenario=args[0]['graph_scenario'] 
        else:
            scenario=None
        if click is not None and click.get("properties") is not None:
            if type.split("_")[0] =='bars':
                data=[extrafunctions.load_bar_id_values("data/cvs_files", 
                                                   click["properties"]["id"],
                                                   "HARWT",
                                                   variety_values,
                                                   HISTORICAL_VALUES ,
                                                   PROJECTED_VALUES,
                                                   planting_values,
                                                   scenario,
                                                   type.split("_")[0]

                                                   )
                                                   ]
            else:
                data=[extrafunctions.load_bar_id_values("data/cvs_files", 
                                                   click["properties"]["id"],
                                                   "HARWT",
                                                   variety_values,
                                                   HISTORICAL_VALUES ,
                                                   PROJECTED_VALUES,
                                                   planting_values,
                                                   scenario,
                                                   type

                                                   )
                                                   ]
            fig = pgo.Figure()

            #min_size = 10
            #max_size = 50

            for trace in data:
                periods = trace["x_axis"]
                hist_val = None
                # Light-blue background for the historical period (first bar)
                fig.add_vrect(
                    x0=-0.5, x1=0.5,         
                    fillcolor="rgba(173, 216, 230, 0.5)",
                    layer="below",
                    line_width=0,
                )
                # Light-red background for projected periods
                fig.add_vrect(
                    x0=0.5, x1=len(periods) - 0.5,   
                    fillcolor="rgba(255, 182, 193, 0.5)",
                    layer="below",
                    line_width=0,
                )
                if type.split("_")[0] == "bars":
                    
                    colorbar=extrafunctions.color_scale("tab20")
                    
                    for i, (model, y_values) in enumerate(trace["models"].items()):
                        if model == "Historical":
                            # Save historical value for the reference dashed line
                            hist_val = y_values[0] if y_values else None
                        else:
                            fig.add_trace(pgo.Bar(
                                x=periods,
                                y=y_values,
                                name=model,
                                marker_color=colorbar[i % len(colorbar)] 
                            ))

                    # Render historical bar and dashed reference line on top
                    if hist_val is not None:
                        # Black bar for the historical period only
                        fig.add_trace(pgo.Bar(
                            x=[periods[0]],
                            y=[hist_val],
                            name="Historical",
                            marker_color="black",
                            legendgroup="historical",       
                            showlegend=True,
                        ))
                        fig.add_trace(pgo.Scatter(
                            x=[periods[1], periods[-1]],
                            y=[hist_val, hist_val],
                            mode="lines",
                            line=dict(dash="dash", color="black", width=3), #['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
                            name="Historical",
                            legendgroup="historical", 
                            showlegend=False,  
                            hovertemplate=f"Historical : {hist_val:,.0f}<extra></extra>",
                            #showlegend=True,
                        ))

                    fig.update_layout(barmode=type.split("_")[1], 
                                      legend=dict(
                                                orientation="h",        
                                                yanchor="bottom",
                                                y=-0.25,                 
                                                xanchor="left",
                                                x=0
                                            ),

                                      )
                elif type in ["lines","markers","lines+markers"]:
                    # Render one trace per model; Historical is shown as a
                    # diamond marker at its period plus a dashed horizontal
                    # reference line across all projected periods.
                    models = trace['models']
                    for name, values in models.items():
                        is_historical = name == 'Historical'
                        x_dup, y_dup = duplicate_points(periods, values)

                        if is_historical:
                            hist_val = next((v for v in values if v is not None), None)

                            if hist_val is not None:
                                # Diamond marker at the historical data point
                                marker_y = values  # [hist_val, None, None, ...]
                                fig.add_trace(pgo.Scatter(
                                    x=periods,
                                    y=marker_y,
                                    mode="markers",
                                    name="Historical",
                                    legendgroup="historical",
                                    showlegend=True,
                                    connectgaps=False,
                                    marker=dict(
                                        color="black",
                                        size=8,
                                        symbol="diamond",
                                    ),
                                ))

                                # Dashed reference line across projected periods
                                # (None where the real data point was, hist_val elsewhere)
                                dash_y = [None if v is not None else hist_val for v in values]
                                fig.add_trace(pgo.Scatter(
                                    x=periods,
                                    y=dash_y,
                                    mode="lines",
                                    name="Historical",
                                    legendgroup="historical",       # mismo grupo
                                    showlegend=False,               # solo quiero una leyenda
                                    connectgaps=False,
                                    line=dict(
                                        dash="dash",
                                        color="black",
                                        width=2,
                                    ),
                                    hovertemplate=f"Historical : {hist_val:,.0f}<extra></extra>",
                                ))

                        else:
                            fig.add_trace(pgo.Scatter(
                            x=periods,
                            y=values,
                            mode=type,
                            name=name,
                            connectgaps=False,
                        ))
                    fig.update_layout(
                        xaxis=dict(
                            title='Period',
                            tickvals=list(range(len(periods))),  # posiciones 0..6
                            ticktext=periods,                    # etiquetas originales
                            tickangle=-30,
                            showspikes=False,  
                        ),
                        yaxis=dict(title='Value', tickformat=',.0f'),
                        hovermode='x unified',
                        legend=dict(orientation='h', y=-0.3),
                    )
                    fig.update_xaxes(type="category")
                elif type in ["markers-scale"]:
                    # Bubble chart: marker size and color both encode the y-value
                    fig.add_trace(pgo.Scatter(x=trace["x"], y=trace["y"],mode="markers", name=click["properties"]["NAME"],
                                              marker=dict(
                                                    size=trace["y"],         # tamaño variable
                                                    sizemode="area",    # el tamaño corresponde al área
                                                    sizeref=2.*max(trace["y"])/(40.**2),  # escala del tamaño (opcional)
                                                    sizemin=4,           # tamaño mínimo
                                                    color=trace["y"],   # color según valor
                                                    colorscale=layout.colorscale,
                                                    showscale=True
                                                ) 
                                              ))

            fig.update_layout(
                title=f"{click['properties']['NAME']} , {click['properties']['STATE_NAME']}:  {next((m['label'] for m in layout.VARIETY if m['value'] == variety_values), variety_values)} variety",
                xaxis_title="",
                yaxis_title="HARWT",
            )
            county_info=(f"{click['properties']['STATE_NAME']}_"
                         f"{click['properties']['NAME']}({click['properties']['id']})_"
                         f"{variety_values}_"
                         f"{planting_values}_"
                         f"{scenario}"
                        )
                    

        return fig, data, county_info

    # -------------------------------------------------
    # Callback: Map variable description panel
    # Generates a human-readable description for the
    # currently selected map variable (data_type).
    # For anomaly types (mean_change, percentage_change,
    # direction_change, stress_simple, yield_change_index)
    # it also renders the equation used to compute the
    # displayed values.
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
                html.Small("Mean change shows how the future average differs from the dataset2 average. Positive " \
                            "values indicate an increase; negative values indicate a decrease relative " \
                            "to the dataset2 period."),
                html.Br(),
                html.B("Equation used: "),
                html.Small([
                    "mean",
                    html.Sub("dataset1"),
                    " − mean",
                    html.Sub("dataset2"),
                ])
                ]
            )
        elif variable == "percentage_change":
            description.extend(
                [
                html.Small(
                    "Percentage change shows the relative difference between the dataset1 and dataset2 averages. "\
                    "Positive values indicate an increase; negative values indicate a decrease relative to the dataset2 period."
                ),
                html.Br(),
                html.B("Equation used: "),
                html.Small([
                        "(",
                        "mean",
                        html.Sub("dataset1"),
                        " − ",
                        "mean",
                        html.Sub("dataset2"),
                        ") / ",
                        "mean",
                        html.Sub("dataset2"),
                        " × 100",
                    ])
                ]
            )
        elif variable == "direction_change":
            description.extend(
                [
                html.Small(
                    "Direction change indicates whether the variable increases, decreases, or stays the same. "\
                    "A value of +1 indicates an increase, 0 indicates no change, and -1 indicates a decrease relative to the dataset2 period."
                ),
                html.Br(),
                html.B("Equation used: "),
                html.Small([
                        "sign(",
                        "mean",
                        html.Sub("dataset1"),
                        " − ",
                        "mean",
                        html.Sub("dataset2"),
                        ")"
                    ])
                ]
            )
        elif variable == "stress_simple":
            description.extend(
                [
                html.Small(
                    "Stress index shows how much the future deviates from dataset2 conditions. "\
                    "Values near -1 indicate extreme stress, 0 indicates neutral conditions, "\
                    "and values near 1 indicate extreme benefit relative to the dataset2 period."
                ),
                html.Br(),
                html.B("Equation used: "),
                html.Small([
                        "1 − (",
                        "mean",
                        html.Sub("dataset1"),
                        " / ",
                        "mean",
                        html.Sub("dataset2"),
                        ")"
                    ])
                ]
            )
        elif variable == "yield_change_index":
            description.extend(
                [
                html.Small(
                    "Yield Change Index (YCI%) shows the relative change in yield compared to dataset2 conditions. "\
                    "Positive values indicate increased yield, negative values indicate decreased yield relative to the dataset2 period."
                ),
                html.Br(),
                html.B("Equation used: "),
                html.Small([
                    "(",
                    "mean",
                    html.Sub("dataset1"),
                    " / ",
                    "mean",
                    html.Sub("dataset2"),
                    " − 1) × 100"
                ])
                ]
            )

        return description
    # -------------------------------------------------
    # Callback: Map title generation
    # Builds the map title string from the currently
    # active dataset inputs. For historical/projected
    # views a single-dataset title is used; for anomaly
    # or direction-change views a dual-dataset
    # comparison title is shown.
    # -------------------------------------------------
    multi_datasets = { 'model':[ Input("dataset1", "value"), Input("dataset2", "value")], 
                     'variety':[ Input("variety", "value"), Input("variety", "value")],
                    'planting':[ Input("dataset1_planting", "value"), Input("dataset2_planting", "value")] ,
                    'scenario':[ Input("dataset1_scenario", "value"), Input("dataset2_scenario", "value")] ,
                'period_years':[ Input("dataset1_years", "value"), Input("dataset2_years", "value")] 
                      
                      }
    single_dataset = {
                      'model':Input("model", "value"),
                    'variety':Input("variety", "value"),
                   'planting':Input("planting", "value"),
                   'scenario':Input("scenario", "value"),
               'period_years':Input("period_years", "value"),
                        }
    @APP.callback(
        Output("map_title", "children"),  # Actualiza el texto del título del mapa
        single_dataset, # args[0]
        Input("data_type", "value"), # args[1]
        multi_datasets, # args[2]
        Input("variable","value") # arg[3]

    )
    def write_map_title(*args):
        single_dataset=args[0]
        data_type=args[1]
        multi_datasets = args[2]
        variable=args[3]

        if data_type in ['historical','projected']:
            variety=single_dataset['variety']
            model=single_dataset['model']
            planting=single_dataset['planting']
            scenario=single_dataset['scenario']
            target_value=single_dataset['period_years']
        else:
            variety=multi_datasets['variety']
            model=multi_datasets['model']
            planting=multi_datasets['planting']
            scenario=multi_datasets['scenario']
            target_value=multi_datasets['period_years']

        if data_type not in ['historical','projected']:
            top_title= (
                    f"HARWT {data_type.replace('_', ' ').capitalize()} variety={variety[0]}",
                    html.Br(),
                    f"{model[0]}_{planting[0]}_{scenario[0]}_{target_value[0]} - "
                    f"{model[1]}_{planting[1]}_{scenario[1]}_{target_value[1]}"
                )
        else:
            top_title=f"{model} {variable} change from {variety} for {target_value}"
        # Construye un título resumido del mapa mostrando periodo, escenario, modelo, variable y referencia
        return (top_title)

    # -------------------------------------------------
    # Callback: Combine anomaly period values
    # Merges dataset1 and dataset2 year selections into
    # a single list stored in a dcc.Store, so other
    # callbacks can consume both values at once.
    # -------------------------------------------------
    @APP.callback(
        Output("anomaly_period_values", "data"),
        [Input("dataset1_years", "value"), Input("dataset2_years", "value")]
    )
    def combine_inputs(dataset1, dataset2):
        return [dataset1, dataset2]
    # -------------------------------------------------
    # Callback: Load CSV and update GeoJSON map layer
    # Reads the appropriate CSV file based on the
    # current dataset selections, computes value
    # classes and a color scale via prepare_data(), and
    # returns updated GeoJSON data, hideout styling
    # props, and a rebuilt categorical colorbar legend.
    # -------------------------------------------------
    multi_datasets = { 'model':[ Input("dataset1", "value"), Input("dataset2", "value")], 
                     'variety':[ Input("variety", "value"), Input("variety", "value")],
                    'planting':[ Input("dataset1_planting", "value"), Input("dataset2_planting", "value")] ,
                    'scenario':[ Input("dataset1_scenario", "value"), Input("dataset2_scenario", "value")] ,
                'period_years':[ Input("dataset1_years", "value"), Input("dataset2_years", "value")] 
                      
                      }
    single_dataset = {
                      'model':Input("model", "value"),
                    'variety':Input("variety", "value"),
                   'planting':Input("planting", "value"),
                   'scenario':Input("scenario", "value"),
               'period_years':Input("period_years", "value"),
                        }
    @APP.callback(
        Output("geojson", "data"),
        Output("geojson", "hideout"),
        Output("colorbar", "children"),
        single_dataset, # args[0]
        Input("data_type", "value"), # args[1]
        multi_datasets # args[2]
    )
    def load_csv_and_update(*args):
        single_dataset=args[0]
        data_type=args[1]
        multi_datasets = args[2]

        if data_type in ['historical','projected']:
            variety=single_dataset['variety']
            model=single_dataset['model']
            planting=single_dataset['planting']
            scenario=single_dataset['scenario']
            target_value=single_dataset['period_years']
        else:
            variety=multi_datasets['variety']
            model=multi_datasets['model']
            planting=multi_datasets['planting']
            scenario=multi_datasets['scenario']
            target_value=multi_datasets['period_years']

        data_filtered,new_classes,colorscale=extrafunctions.prepare_data(variety,model,planting,scenario,target_value,data_type)
        hideout = dict(colorscale=colorscale, classes=new_classes, style=layout.style_default, colorProp="HARWT")

        # Build colorbar title
        if data_type not in ['historical','projected']:
            title= (
                    f"HARWT {data_type.replace('_', ' ').capitalize()} variety={variety}\n"
                    f"{model[0]}_{planting[0]}_{scenario[0]}_{target_value[0]} vs "
                    f"{model[1]}_{planting[1]}_{scenario[1]}_{target_value[1]}"
                )
        else:
            title=f"HARWT variety={variety} target={target_value}"

        # Build categorical colorbar labels based on data type
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

        # NOTE: 'key' is intentionally not changed so the GeoJSON layer
        # is updated in-place rather than destroyed and re-mounted.
        return data_filtered, hideout, new_colorbar
    
    # -------------------------------------------------
    # Callbacks: Border overlay flicker fix
    # When the GeoJSON data updates, the county border
    # overlay is briefly toggled off and then
    # immediately turned back on. This forces a visual
    # refresh of the overlay layer without requiring
    # user interaction.
    # -------------------------------------------------
    @APP.callback(
        Output("overlay-borders", "checked"),
        Input("geojson", "data"),  # <- this wait until the geajon is ready
        prevent_initial_call=True
    )
    def toggle_borders_after_update(_):
        # Step 1: turn borders off to trigger a re-render
        return False

    @APP.callback(
        Output("overlay-borders", "checked", allow_duplicate=True),
        Input("overlay-borders", "checked"),
        prevent_initial_call=True
    )
    def turn_borders_on(checked):
        # Step 2: immediately turn borders back on after they were switched off
        if checked is False:
            return True
        return dash.no_update
    
    # -------------------------------------------------
    # Callback: Hover info panel
    # Updates the floating info box with feature
    # properties when the user hovers over a county
    # on the map. Clears the panel when no feature is
    # hovered.
    # -------------------------------------------------
    @APP.callback(
        Output("info", "children"),
        Input("geojson", "hoverData"),
    )
    def update_info(hover):
        ctx = dash.callback_context
        if not ctx.triggered:
            return layout.get_info()
        prop_id = ctx.triggered[0]["prop_id"]
        if prop_id == "geojson.hoverData":
            feature = hover
        else:
            feature = None
        return layout.get_info(feature)
    