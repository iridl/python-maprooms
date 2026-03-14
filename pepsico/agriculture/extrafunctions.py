import pandas as pd
import glob
import math 
from . import layout 
import numpy as np

#path="data/cvs_files"



def prepare_data(variety, model, planting, scenario, target_value, data_type, path="data/cvs_files"):

    print("params:", variety, target_value, data_type)
    # anom_period[0] = historical data
    # anom_period[1] = forecast data
    if data_type in ["mean_change",
                     "percentage_change",
                     "direction_change",
                     "yield_change_index",
                     "stress_simple"
                     ]:
        if model[0]=='historical':
            csv_file =f"{path}/HARWT_ID_HistBL_PDhist_{variety[0]}_{target_value[0]}_US_CA.csv"
        else:
            csv_file = f"{path}/HARWT_ID_{model[0]}_{scenario[0]}_{planting[0]}_{variety[0]}_{target_value[0]}_US_CA.csv"
        df = pd.read_csv(csv_file,dtype={"ID": str})
        df = df.rename(columns={"ID": "id"})
        # if "CCSUID" in df.columns:
        #     df["id"] = df["CCSUID"].astype(str)
        # elif "FIPS" in df.columns:
        #     df["id"] = df["FIPS"].astype(str)
        df["HARWT"] = pd.to_numeric(df["HARWT"], errors="coerce")

        if model[1]=='historical':
            csv_file_data_aux =f"{path}/HARWT_ID_HistBL_PDhist_{variety[1]}_{target_value[1]}_US_CA.csv"
        else:
            csv_file_data_aux = f"{path}/HARWT_ID_{model[1]}_{scenario[1]}_{planting[1]}_{variety[1]}_{target_value[1]}_US_CA.csv"
        df_fcst = pd.read_csv(csv_file_data_aux,dtype={"ID": str})
        df_fcst = df_fcst.rename(columns={"ID": "id"})

        # if "CCSUID" in df_fcst.columns:
        #     df_fcst["id"] = df_fcst["CCSUID"].astype(str)
        # elif "FIPS" in df_fcst.columns:
        #     df_fcst["id"] = df_fcst["FIPS"].astype(str)
        df_fcst["HARWT"] = pd.to_numeric(df_fcst["HARWT"], errors="coerce")
        #var_map = df.set_index("id")["HARWT"].to_dict()
        # ---- This waranty the sub by id even if the csv is desogarnized 
        if data_type == "mean_change": # (mean_fcst - mean_hist) 
            df = (
                df.merge(df_fcst[["id", "HARWT"]], on="id", how="left", suffixes=("", "_aux"))
                .assign(HARWT=lambda x: round(x["HARWT_aux"] - x["HARWT"]))
                .drop(columns="HARWT_aux")          
                )
        elif data_type == "percentage_change": # (mean_fcst - mean_hist) / mean_hist * 100
            df = (
                df.merge(df_fcst[["id", "HARWT"]], on="id", how="left", suffixes=("", "_aux"))
                .assign(HARWT=lambda x: round((x["HARWT_aux"] - x["HARWT"])/x["HARWT"]*100))
                .drop(columns="HARWT_aux")          
                )
        elif data_type == "yield_change_index": #YCI = mean_fcst / mean_hist ; 
                                                #YCI_% = (mean_fcst / mean_hist − 1) × 100
            df = (
                df.merge(df_fcst[["id", "HARWT"]], on="id", how="left", suffixes=("", "_aux"))
                .assign(HARWT=lambda x: round((x["HARWT_aux"] / x["HARWT"] -1) * 100))
                .drop(columns="HARWT_aux")          
                )
        elif data_type == "stress_simple": #1 − (mean_fcst / mean_hist) ; -1.0  estrés extremo , 0 neutro , 1.0  beneficio extremo
            df = (
                df.merge(df_fcst[["id", "HARWT"]], on="id", how="left", suffixes=("", "_aux"))
                .assign(HARWT=lambda x: round(1-(x["HARWT_aux"] / x["HARWT"])))
                .drop(columns="HARWT_aux")          
                )
        elif data_type == "direction_change":
            df = (
                df.merge(df_fcst[["id", "HARWT"]], on="id", how="left", suffixes=("", "_aux"))
                .assign(HARWT=lambda x: np.sign(x["HARWT_aux"] - x["HARWT"]))
                .drop(columns="HARWT_aux")          
                )
    else:
        if data_type in ["projected"]:
            csv_file = f"{path}/HARWT_ID_{model}_{scenario}_{planting}_{variety}_{target_value}_US_CA.csv"
        else:
            csv_file = f"{path}/HARWT_ID_HistBL_PDhist_{variety}_{target_value}_US_CA.csv"
        print(csv_file)

        df = pd.read_csv(csv_file,dtype={"ID": str})
        df = df.rename(columns={"ID": "id"})
        #print(df[df["FIPS"].astype(str).str.contains("6093")]["FIPS"]) 
        # if "CCSUID" in df.columns:
        #     df["id"] = df["CCSUID"].astype(str)
        # elif "FIPS" in df.columns:
        #     df["id"] = df["FIPS"].astype(str)
        df["HARWT"] = pd.to_numeric(df["HARWT"], errors="coerce")

        #print(df["id"])
        #print(df[df["id"].astype(str).str.contains("6093")]["id"])
        
    var_map = df.set_index("id")["HARWT"].to_dict()
    
    #print(file)
        


    # Actualizar geojson (solo data, no reemplazar GeoJSON completo)
    new_features = []
    for feature in layout.data["features"]:
        fid = str(feature["properties"].get("id", ""))
        feature["properties"]["HARWT"] = var_map.get(fid, np.nan)
        new_features.append(feature)

    data_filtered = {
        "type": "FeatureCollection",
        "features": [f for f in new_features if f["properties"].get("HARWT") is not None and not np.isnan(f["properties"].get("HARWT"))]
    }

    # Calcular nuevas clases
    if data_type in ["percentage_change","yield_change_index"]:
        min_val = -100
        max_val = 100
    elif data_type in ["direction_change","stress_simple"]:
        min_val = -1
        max_val = 1
    else:
        min_val = df["HARWT"].min()
        max_val = df["HARWT"].max()

    #print(f"{min_val} {max_val}")

    base = calc_base(min_val, max_val)
    if data_type in ["direction_change","stress_simple"]:
        new_classes = gen_clases(min_val, max_val, base=0.25)
    elif data_type in ["percentage_change","yield_change_index"]:
        new_classes = gen_clases(min_val, max_val, base=25)
    else:
        new_classes = gen_clases(min_val, max_val, base=base)

    print(new_classes)

    colorscale=color_scale(data_type)

    return data_filtered,new_classes,colorscale 

def calc_base(min_val, max_val, n=9):
    rango = max_val - min_val
    step_ideal = rango / (n - 1)
    exp = math.floor(math.log10(step_ideal))
    base = 10 ** exp
    multiplo = round(step_ideal / base)
    base = multiplo * base
    return max(base, 1)

def gen_clases(min_val, max_val, n=9, base=1000):
    min_val = int(min_val)
    max_val = int(max_val)
    lo = (min_val // base) * base
    hi = (max_val // base) * base
    if hi <= lo:
        return [lo + i * base for i in range(n)]
    raw_step = (hi - lo) / (n - 1)
    clases = [int(round(lo + raw_step * i) // base * base) for i in range(n)]
    for i in range(1, n):
        if clases[i] <= clases[i-1]:
            clases[i] = clases[i-1] + base
    clases[-1] = hi
    for i in range(n-2, -1, -1):
        if clases[i] >= clases[i+1]:
            clases[i] = clases[i+1] - base
    if clases[0] < lo:
        return [lo + i * base for i in range(n)]
    #clases.append(clases[-1]+0.1)
    return clases


def color_scale(input):
    if input == "mean_change":
        colorscale = [
                        "#8C510A", "#BF812D", "#DFC27D", "#F6E8C3",   # déficit (marrones)
                        "#F7F7F7",                                          # normal
                        "#D9F0D3", "#A6DBA0", "#5AAE61", "#1B7837"    # excedente (verdes)
                    ]
    elif input == "direction_change":
        colorscale = [
                        "#B2182B", "#B2182B", "#B2182B", "#B2182B",  # 4 Disminución leve
                        "#F7F7F7",  
                        "#2166AC", "#2166AC", "#2166AC", "#2166AC",  # 9 Muy fuerte aumento
                    ]
    elif input in ['yield_change_index','stress_simple']: 
        colorscale = [
                        "#7F0000", "#B30000",  "#D7301F",  "#FC8D59",  
                        "#FFFFFF",  # blanco (sin cambio)
                        "#91CF60", "#4DAC26",  "#238B45",  "#00441B",  # verde muy oscuro (ganancia severa)
                    ]
        
    elif input == 'stress_simple_invert': 
        colorscale =  [
    "#67001F",  # -1.0  estrés extremo
    "#B2182B",  # alto estrés
    "#D6604D",  # estrés moderado
    "#F4A582",  # estrés leve
    "#FFFFFF",  #  0.0  neutro
    "#D9F0D3",  # beneficio leve
    "#7FBF7B",  # beneficio moderado
    "#35978F",  # beneficio alto
    "#01665E",  #  1.0  beneficio extremo
]
    elif input == "percentage_change":
        colorscale = [
                        "#67001F", "#B2182B", "#D6604D", "#F4A582",  # 4 Disminución leve
                        "#F7F7F7",  
                        "#92C5DE", "#4393C3", "#2166AC", "#053061",  # 9 Muy fuerte aumento
                    ]
    else: 
        colorscale = [
                        "#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C",
                        "#FD6F33",
                        "#FC4E2A", "#E31A1C", "#BD0026", "#800026"
                        ]
    return colorscale

def cargar_valores_id(path_folder, id_buscar,var,variety_values,hist_years_values,fcst_years_values):
    csv_files = sorted(glob.glob(f"{path_folder}/*.csv"))

    print(variety_values)

    nombres_archivos = []
    valores = []
    names = []


    for file in csv_files:
        df = pd.read_csv(file,dtype={"FIPS": str})

        # verificar que exista la columna ID
        if "FIPS" not in df.columns:
            continue

        df["FIPS"] = df["FIPS"].astype(str).str.strip()
        id_buscar = str(id_buscar).strip()

        fila = df[df["FIPS"] == id_buscar]
        #print('FIPS son')
        #print(df["FIPS"])


        if fila.empty:
            continue

        # Asumimos que la columna del valor se llama Var
        valor = fila[var].iloc[0]
        #name = fila["NAME"].iloc[0]

        # Nombre corto del archivo (sin ruta)
        nombre = "_".join(file.split("/")[-1].split("_")[-2:]).replace(".csv","")
        nombress = next((x for x in variety_values if x in file), None)
        #print(nombress)
        if next((x for x in hist_years_values if x in file), None): 
            data_source=f'Historical {next((x for x in hist_years_values if x in file), None)}'
        elif next((x for x in fcst_years_values if x in file), None): 
            data_source=f'Projected {next((x for x in fcst_years_values if x in file), None)}'

        print(data_source)

        nombres_archivos.append(nombre)
        valores.append(valor)
        names.append(data_source)

    # Trace compatible con dcc.Graph
    trace = {
        "x": nombres_archivos,
        "y": valores,
      #  "type": type,
      #  "name": f"Valores para {id_buscar}"
    }

    return trace
