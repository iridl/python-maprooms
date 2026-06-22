import pandas as pd
import glob
import math 
from . import layout 
import numpy as np
import os
import bcrypt


def hash_user_password(user: str, password: str) -> str:
    hashed = bcrypt.hashpw(
        password.encode(),
        bcrypt.gensalt()
    ).decode()

    return f"{user}:{hashed}"

def load_valid_users_hash(filepath=".users") -> dict:
    """Loads users with hashed passwords from a file."""
    users = {}
    with open(filepath, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line and ":" in line:
                user, hashed = line.split(":", 1)
                users[user.strip()] = hashed.strip()
    return users

def verify_password(plain: str, hashed: str) -> bool:
    """Verifies a password (plaintext received from the popup) against its bcrypt hash."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def load_valid_users() -> dict:
    """
    Reads users from the .users 
    Expected format in .users:
        AUTH_USERS=admin:password123,usuario2:otraClave456
    """
    env_path = os.path.join(os.path.dirname(__file__), ".users")

    if not os.path.exists(env_path):
        raise FileNotFoundError(
            f"File not found .users in: {env_path}"
        )

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("AUTH_USERS="):
                raw = line.split("=", 1)[1]
                return dict(
                    pair.split(":", 1)
                    for pair in raw.split(",")
                )

    raise ValueError(
        "Unauthorized user in AUTH_USERS \n"
    )

def prepare_data(variety, model, planting, scenario, target_value, data_type, path="data/cvs_files"):

    #by default forescast is on the first ([dataset 1]) position and historical is on second ([dataset 2]) position 
    if data_type in ["mean_change",
                     "percentage_change",
                     "direction_change",
                     "yield_change_index",
                     "stress_simple"
                     ]:
        if model[1]=='historical':
            csv_file =f"{path}/HARWT_ID_HistBL_PDhist_{variety[1]}_{target_value[1]}_US_CA.csv"
        else:
            csv_file = f"{path}/HARWT_ID_{model[1]}_{scenario[1]}_{planting[1]}_{variety[1]}_{target_value[1]}_US_CA.csv"
        df = pd.read_csv(csv_file,dtype={"ID": str})
        df = df.rename(columns={"ID": "id"})

        df["HARWT"] = pd.to_numeric(df["HARWT"], errors="coerce")

        if model[0]=='historical':
            csv_file_data_aux =f"{path}/HARWT_ID_HistBL_PDhist_{variety[0]}_{target_value[0]}_US_CA.csv"
        else:
            csv_file_data_aux = f"{path}/HARWT_ID_{model[0]}_{scenario[0]}_{planting[0]}_{variety[0]}_{target_value[0]}_US_CA.csv"
        df_fcst = pd.read_csv(csv_file_data_aux,dtype={"ID": str})
        df_fcst = df_fcst.rename(columns={"ID": "id"})

        df_fcst["HARWT"] = pd.to_numeric(df_fcst["HARWT"], errors="coerce")

        # ---- This waranty the sub by id even if the csv is desogarnized 
        if data_type == "mean_change": # (mean_dataset1 - mean_dataset2) 
            df = (
                df.merge(df_fcst[["id", "HARWT"]], on="id", how="left", suffixes=("", "_aux"))
                .assign(HARWT=lambda x: round(x["HARWT_aux"] - x["HARWT"]))
                .drop(columns="HARWT_aux")          
                )
        elif data_type == "percentage_change": # (mean_dataset1 - mean_dataset2) / mean_dataset2 * 100
            df = (
                df.merge(df_fcst[["id", "HARWT"]], on="id", how="left", suffixes=("", "_aux"))
                .assign(HARWT=lambda x: round((x["HARWT_aux"] - x["HARWT"])/x["HARWT"]*100))
                .drop(columns="HARWT_aux")          
                )
        elif data_type == "yield_change_index": #YCI = mean_dataset1 / mean_dataset2 ; 
                                                #YCI_% = (mean_dataset1 / mean_dataset2 − 1) × 100
            df = (
                df.merge(df_fcst[["id", "HARWT"]], on="id", how="left", suffixes=("", "_aux"))
                .assign(HARWT=lambda x: round((x["HARWT_aux"] / x["HARWT"] -1) * 100))
                .drop(columns="HARWT_aux")          
                )
        elif data_type == "stress_simple": #1 − (mean_dataset1 / mean_dataset2) ; -1.0  estrés extremo , 0 neutro , 1.0  beneficio extremo
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

        df = pd.read_csv(csv_file,dtype={"ID": str})
        df = df.rename(columns={"ID": "id"})
        df["HARWT"] = pd.to_numeric(df["HARWT"], errors="coerce")
        
    var_map = df.set_index("id")["HARWT"].to_dict()
    
    # Update geojson (only data, not replace the whole GeoJSON )
    new_features = []
    for feature in layout.data["features"]:
        fid = str(feature["properties"].get("id", ""))
        feature["properties"]["HARWT"] = var_map.get(fid, np.nan)
        new_features.append(feature)

    data_filtered = {
        "type": "FeatureCollection",
        "features": [f for f in new_features if f["properties"].get("HARWT") is not None and not np.isnan(f["properties"].get("HARWT"))]
    }

    # Data range 
    if data_type in ["percentage_change","yield_change_index"]:
        min_val = -100
        max_val = 100
    elif data_type in ["direction_change","stress_simple"]:
        min_val = -1
        max_val = 1
    else:
        min_val = df["HARWT"].min()
        max_val = df["HARWT"].max()

    base = calc_base(min_val, max_val)
    if data_type in ["direction_change","stress_simple"]:
        new_classes = gen_color_class(min_val, max_val, base=0.25)
    elif data_type in ["percentage_change","yield_change_index"]:
        new_classes = gen_color_class(min_val, max_val, base=25)
    else:
        new_classes = gen_color_class(min_val, max_val, base=base)

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

def gen_color_class(min_val, max_val, n=9, base=1000):
    min_val = int(min_val)
    max_val = int(max_val)
    lo = (min_val // base) * base
    hi = (max_val // base) * base
    if hi <= lo:
        return [lo + i * base for i in range(n)]
    raw_step = (hi - lo) / (n - 1)
    color_class = [int(round(lo + raw_step * i) // base * base) for i in range(n)]
    for i in range(1, n):
        if color_class[i] <= color_class[i-1]:
            color_class[i] = color_class[i-1] + base
    color_class[-1] = hi
    for i in range(n-2, -1, -1):
        if color_class[i] >= color_class[i+1]:
            color_class[i] = color_class[i+1] - base
    if color_class[0] < lo:
        return [lo + i * base for i in range(n)]
    return color_class


def color_scale(input):
    if input == "mean_change":
        colorscale = [
                        "#8C510A", "#BF812D", "#DFC27D", "#F6E8C3",   # déficit (browns)
                        "#F7F7F7",                                          # normal
                        "#D9F0D3", "#A6DBA0", "#5AAE61", "#1B7837"    # excedente (greens)
                    ]
    elif input == "direction_change":
        colorscale = [
                        "#B2182B", "#B2182B", "#B2182B", "#B2182B",  # 4 Slight decrease
                        "#F7F7F7",  
                        "#2166AC", "#2166AC", "#2166AC", "#2166AC",  # 9 High increase
                    ]
    elif input in ['yield_change_index','stress_simple']: 
        colorscale = [
                        "#7F0000", "#B30000",  "#D7301F",  "#FC8D59",  
                        "#FFFFFF",  # white (no changes)
                        "#91CF60", "#4DAC26",  "#238B45",  "#00441B",  
                    ]
        
    elif input == 'stress_simple_invert': 
        colorscale =  [
    "#67001F",  # -1.0  extreme stress
    "#B2182B",  # alto estrés
    "#D6604D",  # high stress
    "#F4A582",  # mild stress
    "#FFFFFF",  #  0.0  neutral
    "#D9F0D3",  # slight benefit
    "#7FBF7B",  # moderate benefit
    "#35978F",  # high profit
    "#01665E",  #  1.0  extreme high
]
    elif input == "percentage_change":
        colorscale = [
                        "#67001F", "#B2182B", "#D6604D", "#F4A582",  # 4 Slight decrease
                        "#F7F7F7",  
                        "#92C5DE", "#4393C3", "#2166AC", "#053061",  # 9 High increase
                    ]
    elif input == "tab20":
        colorscale = [
                        "#1f77b4",
                        "#ff7f0e",
                        "#2ca02c",
                        "#d62728",
                        "#9467bd",
                        "#8c564b",
                        "#e377c2",
                        "#7f7f7f",
                        "#bcbd22",
                        "#17becf"
                        ]
    else: 
        colorscale = [
                        "#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C",
                        "#FD6F33",
                        "#FC4E2A", "#E31A1C", "#BD0026", "#800026"
                        ]
    return colorscale

from pathlib import Path

# def cargar_valores_id_bar_original(path_folder, id_search, var, variety_values, hist_years_values, fcst_years_values, planting_values, scenario, type):
#     csv_files = sorted(glob.glob(f"{path_folder}/*.csv"))

#     x_axis = []
#     # Cambia esto: un dict {model: {period: value}}
#     data_by_model = {}

#     for file in csv_files:
#         df = pd.read_csv(file, dtype={"ID": str})

#         if "ID" not in df.columns:
#             continue

#         df["ID"] = df["ID"].astype(str).str.strip()
#         id_search = str(id_search).strip()
#         fila = df[df["ID"] == id_search]

#         if fila.empty:
#             continue
#         elif variety_values not in file:
#             continue
#         elif planting_values not in file:
#             continue
#         elif scenario and scenario not in file and not any(v in file for v in hist_years_values):
#             continue

#         valor = fila[var].iloc[0]

#         if next((x for x in hist_years_values if x in file), None):
#             period = file.split("/")[-1].split("_")[5].replace(".csv", "")
#             dataset_name = 'Historical'
#         elif next((x for x in fcst_years_values if x in file), None):
#             period = file.split("/")[-1].split("_")[6].replace(".csv", "")
#             dataset_name = file.split("/")[-1].split("_")[2].replace(".csv", "")

#         # Acumular en el dict
#         if dataset_name not in data_by_model:
#             data_by_model[dataset_name] = {}
#         data_by_model[dataset_name][period] = valor

#         if period not in x_axis:
#             x_axis.append(period)

#     x_axis = sorted(x_axis)  # ordenar períodos cronológicamente

#     # Armar trace compatible con el nuevo gráfico stacked
#     trace = {
#         "x_axis": x_axis,
#         "models": {}
#     }

#     for model, period_vals in data_by_model.items():
#         # y es una lista ordenada según x_axis, None si falta el período
#         trace["models"][model] = [period_vals.get(p) for p in x_axis]

#     return trace

#this function was optimezed 
def load_bar_id_values( 
    path_folder, id_search, var, variety_values,
    hist_years_values, fcst_years_values, planting_values, scenario, type
):
    id_search = str(id_search).strip()
    hist_set = set(hist_years_values)
    fcst_set = set(fcst_years_values)

    csv_files = sorted(glob.glob(f"{path_folder}/*.csv"))

    # Pre-filter by filename before reading anything.
    def file_passes(file):
        if variety_values not in file:
            return False
        is_hist = any(v in file for v in hist_set)
        if not is_hist and planting_values not in file:
            return False
        if type=='bars' and not is_hist and scenario and scenario not in file:
            return False
        elif type != 'bars' and not is_hist:
            # allow all scenarios
            pass
        return True

    filtered_files = [f for f in csv_files if file_passes(f)]

    data_by_model = {}
    x_axis_set = set()

    for file in filtered_files:
        fname = Path(file).name
        parts = fname.replace(".csv", "").split("_")

        is_hist = any(v in file for v in hist_set)
        is_fcst = any(v in file for v in fcst_set)

        if is_hist:
            period = parts[5]
            dataset_name = "Historical"
        elif is_fcst:
            period = parts[6]
            if type == 'bars':
                dataset_name = parts[2]
            else:
                dataset_name = f"{parts[2]} {parts[3]}" 
        else:
            continue

        # Read only the necessary columns
        try:
            df = pd.read_csv(file, usecols=["ID", var], dtype={"ID": str})
        except ValueError:
            continue  # Column 'var' does not exist in this file.

        df["ID"] = df["ID"].str.strip()
        fila = df[df["ID"] == id_search]

        if fila.empty:
            continue

        valor = fila[var].iloc[0]

        data_by_model.setdefault(dataset_name, {})[period] = valor
        x_axis_set.add(period)

    x_axis = sorted(x_axis_set)

    return {
        "x_axis": x_axis,
        "models": {
            model: [period_vals.get(p) for p in x_axis]
            for model, period_vals in data_by_model.items()
        },
    }

# def cargar_valores_id(path_folder, id_search,var,variety_values,hist_years_values,fcst_years_values,planting_values,scenario,type):
#     csv_files = sorted(glob.glob(f"{path_folder}/*.csv"))

#     #print(variety_values)

#     nombres_archivos = []
#     valores = []
#     names = []
#     x_axis = []
#     models = []


#     for file in csv_files:
#         df = pd.read_csv(file,dtype={"ID": str})

#         # verificar que exista la columna ID
#         if "ID" not in df.columns:
#             continue

#         df["ID"] = df["ID"].astype(str).str.strip()
#         id_search = str(id_search).strip()

#         fila = df[df["ID"] == id_search]
#         #print('FIPS son')
#         #print(f'filtrando para serie {df["ID"]}')



#         if fila.empty:
#             continue
#         elif variety_values not in file:
#             continue
#         elif planting_values not in file:
#             continue 
#         elif scenario and scenario not in file and not any(v in file for v in hist_years_values):
#             continue


#         # Asumimos que la columna del valor se llama Var
#         valor = fila[var].iloc[0]
#         #name = fila["NAME"].iloc[0]

#         # Nombre corto del archivo (sin ruta)
#         # HARWT_ID_CanESM2.CRCM5-UQAM_rcp85_PDy0n30_Atlantic_2021-2025_US_CA.csv
#         #  0    1         2             3      4       5        6       7 8
#         #
#         #
#         # HARWT_ID_HistBL_PDhist_Atlantic_2006-2010_US_CA.csv
#         #.  0    1   2      3       4         5      6  7  
#         #nombre = "_".join(file.split("/")[-1].split("_")[-2:]).replace(".csv","")
        
#         nombress = next((x for x in variety_values if x in file), None)
#         #print(f"Nombre de archivo es {file}")
#         if next((x for x in hist_years_values if x in file), None): 
#             data_source=f'Historical {next((x for x in hist_years_values if x in file), None)}'
#             nombre = "_".join([file.split("/")[-1].split("_")[i] for i in [5]]).replace(".csv","")
#             period=file.split("/")[-1].split("_")[5].replace(".csv","")
#             nombre="Hitorical_"+nombre
#             dataset_name='Historical'
#         elif next((x for x in fcst_years_values if x in file), None): 
#             #print(f"archivo entrando {file}")
#             data_source=f'Projected {next((x for x in fcst_years_values if x in file), None)}'
#             if scenario:
#                 nombre = "_".join([file.split("/")[-1].split("_")[i] for i in [2,6]]).replace(".csv","")
#             else:
#                 nombre = "_".join([file.split("/")[-1].split("_")[i] for i in [2,3,6]]).replace(".csv","")
#             period=file.split("/")[-1].split("_")[6].replace(".csv","")
#             dataset_name=file.split("/")[-1].split("_")[2].replace(".csv","")

#         #print(f"Data source es {data_source}")

#         nombres_archivos.append(nombre)
#         valores.append(valor)
#         if period not in x_axis:
#             x_axis.append(period)
#         if dataset_name not in  models :
#             models.append(dataset_name)
#         #names.append(data_source)

#     # Trace compatible con dcc.Graph
#     #print(f"Valores es {valores}")
#     if type=='bars':
#         trace = {
#             "x":  x_axis,
#             "y": valores,
#             "name": models
#         }
#     else:
#         trace = {
#             "x": nombres_archivos,
#             "y": valores,
#         #  "type": type,
#         #  "name": f"Valores para {id_search}"
#         }

#     return trace
