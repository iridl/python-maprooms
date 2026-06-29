import pandas as pd
import glob
import math 
from . import layout 
import numpy as np
import os
import bcrypt
from pathlib import Path

def hash_user_password(user: str, password: str) -> str:
    """Hash a user's password using bcrypt and return a formatted credential string.

    Parameters
    ----------
    user : str
        The username to associate with the hashed password.
    password : str
        The plaintext password to be hashed.

    Returns
    -------
    str
        A string in the format ``"username:hashed_password"``, ready to be
        stored in the ``.users`` credentials file.
    """
    hashed = bcrypt.hashpw(
        password.encode(),
        bcrypt.gensalt()
    ).decode()

    return f"{user}:{hashed}"

def load_valid_users_hash(filepath=".users") -> dict:
    """Load users and their bcrypt-hashed passwords from a credentials file.

    Each line in the file must follow the format ``username:hashed_password``.
    Blank lines and lines without a colon are silently ignored.

    Parameters
    ----------
    filepath : str, optional
        Path to the credentials file. Defaults to ``".users"``.

    Returns
    -------
    dict
        A dictionary mapping each username (str) to its bcrypt hash (str).
    """
    users = {}
    with open(filepath, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line and ":" in line:
                user, hashed = line.split(":", 1)
                users[user.strip()] = hashed.strip()
    return users

def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash.

    Parameters
    ----------
    plain : str
        The plaintext password received from the user (e.g. from a login popup).
    hashed : str
        The bcrypt-hashed password retrieved from the credentials store.

    Returns
    -------
    bool
        ``True`` if the plaintext password matches the hash, ``False`` otherwise.
    """
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def load_valid_users() -> dict:
    """Load plaintext user credentials from a ``.users`` env-style file.

    The file must be located in the same directory as this module and contain
    a line with the format::

        AUTH_USERS=user1:password1,user2:password2

    Parameters
    ----------
    None

    Returns
    -------
    dict
        A dictionary mapping each username (str) to its plaintext password (str).

    Raises
    ------
    FileNotFoundError
        If the ``.users`` file does not exist in the module's directory.
    ValueError
        If the ``AUTH_USERS`` key is not found in the file.
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
    """Load, process, and merge crop yield CSV data for map visualization.

    Reads one or two CSV files depending on ``data_type``, computes the
    requested transformation between datasets (e.g. percentage change,
    direction of change), injects the resulting HARWT values into the shared
    GeoJSON ``layout.data``, and returns the filtered GeoJSON together with
    the colour-class breakpoints and colour scale appropriate for the metric.

    When ``data_type`` requires a comparison (``mean_change``,
    ``percentage_change``, ``direction_change``, ``yield_change_index``,
    ``stress_simple``), both ``variety``, ``model``, ``planting``,
    ``scenario``, and ``target_value`` must be 2-element sequences where
    index 0 corresponds to the *forecast* dataset and index 1 to the
    *baseline/reference* dataset.  For single-dataset types (``projected``,
    ``historical``), scalar values are expected.

    Parameters
    ----------
    variety : str or list of str
        Crop variety identifier(s).  Two-element list for comparison types.
    model : str or list of str
        Climate model identifier(s) (e.g. ``"CanESM2.CanRCM4"``).  Use
        ``"historical"`` to select the historical baseline file.
        Two-element list for comparison types.
    planting : str or list of str
        Planting identifier(s) used in the CSV filename.
        Two-element list for comparison types (eg. No adaptation -> ``"PDhist"`` ).
    scenario : str or list of str
        Emissions scenario identifier(s) (e.g. ``"rcp45"``).
        Two-element list for comparison types.
    target_value : str or list of str
        Target identifier(s) embedded in the filename.
        Two-element list for comparison types.
    data_type : str
        Transformation to apply.  One of:

        * ``"mean_change"``        – absolute difference (forecast − baseline)
        * ``"percentage_change"``  – relative difference as a percentage
        * ``"yield_change_index"`` – YCI expressed as a percentage: (fcst/base − 1) × 100
        * ``"stress_simple"``      – stress index: 1 − (fcst / base)
        * ``"direction_change"``   – sign of the difference (−1, 0, or +1)
        * ``"projected"``          – raw forecast values (no comparison)
        * ``"historical"``         – raw historical baseline values
    path : str, optional
        Directory containing the CSV files.  Defaults to
        ``"data/cvs_files"``.

    Returns
    -------
    data_filtered : dict
        A GeoJSON ``FeatureCollection`` (dict) with the computed HARWT value
        injected into each feature's ``properties``.  Features whose HARWT
        is ``NaN`` or missing are excluded.
    new_classes : list of int
        A list of 9 breakpoint values used to assign colours to the choropleth
        classes.
    colorscale : list of str
        A list of 9 hex colour strings corresponding to the 9 colour classes.
    """
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
    """Compute a round step size for dividing a numeric range into ``n`` classes.

    Calculates an aesthetically clean interval by rounding the ideal step to
    the nearest power-of-10 multiple, ensuring colour-class breakpoints land
    on human-readable numbers.

    Parameters
    ----------
    min_val : float
        Minimum value of the data range.
    max_val : float
        Maximum value of the data range.
    n : int, optional
        Desired number of colour classes.  Defaults to ``9``.

    Returns
    -------
    base : int or float
        A rounded step size guaranteed to be at least ``1``.
    """
    rango = max_val - min_val
    step_ideal = rango / (n - 1)
    exp = math.floor(math.log10(step_ideal))
    base = 10 ** exp
    multiplo = round(step_ideal / base)
    base = multiplo * base
    return max(base, 1)

def gen_color_class(min_val, max_val, n=9, base=1000):
    """Generate ``n`` evenly spaced, rounded colour-class breakpoints.

    Breakpoints are snapped to multiples of ``base`` so that legend labels
    are clean round numbers.  The algorithm guarantees strict monotonicity:
    each breakpoint is larger than the previous one by at least ``base``.

    Parameters
    ----------
    min_val : float
        Minimum value of the data range.
    max_val : float
        Maximum value of the data range.
    n : int, optional
        Number of breakpoints to generate.  Defaults to ``9``.
    base : float, optional
        Rounding unit.  Breakpoints are multiples of this value.
        Defaults to ``1000``.

    Returns
    -------
    color_class : list of int
        A strictly increasing list of ``n`` integer breakpoints aligned to
        multiples of ``base``.
    """
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
    """Return a 9-colour hex palette suited to the requested data type.

    Each palette is a diverging or sequential scheme chosen to represent the
    sign and magnitude of the underlying metric intuitively (e.g. red for
    deficit/stress, green or blue for surplus/benefit).

    Parameters
    ----------
    input : str
        The data-type identifier.  Recognised values:

        * ``"mean_change"``        – brown-to-green diverging scale
        * ``"direction_change"``   – red / neutral / blue categorical scale
        * ``"yield_change_index"`` – dark-red to dark-green diverging scale
        * ``"stress_simple"``      – dark-red to dark-green diverging scale
        * ``"stress_simple_invert"``– dark-red to teal diverging scale (inverted)
        * ``"percentage_change"``  – red-to-blue diverging scale
        * ``"tab20"``              – qualitative 10-colour Matplotlib tab20 subset
        * any other value          – yellow-to-dark-red sequential scale

    Returns
    -------
    colorscale : list of str
        A list of 9 hex colour strings ordered from the lowest to the
        highest class value.
    """
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

#this function was optimezed 
def load_bar_id_values( 
    path_folder, id_search, var, variety_values,
    hist_years_values, fcst_years_values, planting_values, scenario, type
):
    """Collect time-series values for a single grid-cell ID across all matching CSV files.

        Scans every CSV in ``path_folder``, pre-filters by filename to include
        only files that match the requested variety, year ranges, planting date,
        and (for bar charts) scenario, then reads only the two columns needed
        (``ID`` and ``var``) before extracting the row for ``id_search``.
        Results are grouped by model/dataset name and sorted by period, ready
        to be passed directly to a bar or line chart.

        Parameters
        ----------
        path_folder : str
            Directory containing the CSV files to scan.
        id_search : str or int
            The grid-cell identifier to look up.  Leading/trailing whitespace is
            stripped before comparison.
        var : str
            Name of the data column to extract from each CSV (e.g. ``"HARWT"``).
        variety_values : str
            Crop variety token that must appear in the filename.
        hist_years_values : list of str
            Year tokens that identify historical CSV files.
        fcst_years_values : list of str
            Year tokens that identify projected CSV files.
        planting_values : str
            Planting identifier(s) used in the CSV filename.
        scenario : str or None
            Emissions scenario identifier(s) (e.g. ``"rcp45"``).
        type : str
            Chart type selector.  Use ``"bars"`` to filter by scenario and label
            datasets by model only (necessary because of how the data must be formatted to create bars);
            any other value includes the scenario token in the dataset label.

        Returns
        -------
        dict
            A dictionary with two keys:

            ``"x_axis"`` : list of str
                Sorted list of period tokens found across all matching files.
            ``"models"`` : dict of {str: list}
                Maps each dataset name (model or ``"Historical"``) to a list of
                values aligned to ``x_axis``.  Missing periods are represented
                as ``None``.
        """
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


