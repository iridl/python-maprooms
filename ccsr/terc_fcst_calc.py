from pathlib import Path
import datetime
import predictions
from dateutil.relativedelta import relativedelta
import xarray as xr


def get_issues(fcst_conf):
    files_list = Path(fcst_conf["forecast_path"]).glob(
        f"forecast_{fcst_conf['var']}_*.nc"
    )
    return sorted([datetime.datetime(
        int(f.name[-7:-3]),
        predictions.strftimeb2int(f.name[-10:-7]),
        1,
    ) for f in files_list], reverse=True)


def get_fcst(fcst_conf, start_date=None, lead_time=None):
    data_path = Path(fcst_conf["forecast_path"])
    var = fcst_conf["var"]
    S = None if start_date is None else (start_date[0:3] + start_date[4:8])
    if S is None:
        S = get_issues(fcst_conf)[0].strftime('%b%Y')
    fcst_ds = xr.open_dataset(data_path / f"forecast_{var}_{S}.nc")
    fcst_ds = (
        fcst_ds
        .assign_coords({"cat" : fcst_ds["category"]})
        .assign_coords(longitude=(((fcst_ds.longitude + 180) % 360) - 180))
        .sortby("longitude")
        .rename({"longitude": "X", "latitude": "Y"})
    )
    if lead_time is not None:
        fcst_ds = fcst_ds.sel(lead=int(lead_time))
    return fcst_ds


def get_targets_dict(fcst_ds, S_date):
    leads = fcst_ds["lead"]
    return [
        {
            "label": predictions.target_range_formatting(
                S_date + relativedelta(months=(int(lead))),
                S_date + relativedelta(months=(int(lead) + 2)),
                "months",
            ),
            "value": lead,
        } for lead in leads.values
    ]