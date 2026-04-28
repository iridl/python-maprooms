from pathlib import Path
import datetime
import predictions
from dateutil.relativedelta import relativedelta
import xarray as xr


def get_issues(fcst_conf):
    if fcst_conf["forecast"] == "Seasonal" :
        files_list = Path(fcst_conf["forecast_path"]).glob(
            f"forecast_{fcst_conf['var']}_*.nc"
        )
        issues = sorted([datetime.datetime(
            int(f.name[-7:-3]),
            predictions.strftimeb2int(f.name[-10:-7]),
            1,
        ) for f in files_list], reverse=True)
    else:
        files_list = Path(fcst_conf["forecast_path"]).glob(
            f"MME_Subx_global_{fcst_conf['var']}_Fri_*_wk3_week1234_forecast.nc"
        )
        issues = sorted([datetime.datetime(
            int(f.name[30:34]),
            predictions.strftimeb2int(f.name[26:29]),
            int(f.name[23:25]),
        ) for f in files_list], reverse=True)
    return issues


def get_fcst(fcst_conf, start_date=None, lead_time=None):
    data_path = Path(fcst_conf["forecast_path"])
    var = fcst_conf["var"]
    S = None if start_date is None else predictions.start_to_file_start(start_date)
    if S is None:
        S_format = '%b%Y' if fcst_conf["forecast"] == "Seasonal" else '%d_%b_%Y'
        S = get_issues(fcst_conf)[0].strftime(S_format)
    if fcst_conf["forecast"] == "Seasonal" :
        fcst_file = f"forecast_{fcst_conf['var']}_{S}.nc"
    else:
        fcst_file = (
            f"MME_Subx_global_{fcst_conf['var']}_Fri_{S}_wk3_week1234_forecast.nc"
        )
    fcst_ds = xr.open_dataset(data_path / fcst_file)
    #print(fcst_ds["prob"].data)
    fcst_ds = (
        fcst_ds
        .assign_coords({"cat" : fcst_ds["category"]})
        .assign_coords(longitude=(((fcst_ds.longitude + 180) % 360) - 180))
        .sortby(["longitude", "latitude"])
        .rename({"longitude": "X", "latitude": "Y"})
    )
    if lead_time is not None:
        fcst_ds = fcst_ds.sel(lead=int(lead_time))
    return fcst_ds


def get_targets_dict(fcst_conf, S_date):
    fcst_ds = get_fcst(fcst_conf)
    if fcst_conf["forecast"] == "Seasonal" :
        lead_start = [
            relativedelta(months=(int(lead)))
            for lead in fcst_ds["lead"].values
        ]
        lead_end = [
            relativedelta(months=(int(lead) + 2))
            for lead in fcst_ds["lead"].values
        ]
        time_units = "months"
    else:
        lead_start = [
            relativedelta(days=(1 + 7 * (int(lead) - 1)))
            for lead in fcst_ds["lead"].values
        ]
        lead_end = [
            relativedelta(days=(1 + 7 * (int(lead) - 1) + 6))
            for lead in fcst_ds["lead"].values
        ]
        time_units = "days"
    return [
        {
            "label": predictions.target_range_formatting(
                S_date + lead_start[l], S_date + lead_end[l], time_units,
            ),
            "value": lead,
        } for l, lead in enumerate(fcst_ds["lead"].values)
    ]
