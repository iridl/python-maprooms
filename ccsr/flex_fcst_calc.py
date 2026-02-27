import pycpt
import iri_fd_elr as ife
from pathlib import Path
import pingrid
import numpy as np
import datetime


def get_targets_dict(fcst_conf, fcst_ds, start_date):
    if fcst_conf["method"] == "pycpt":
        return pycpt.pycptv2_targets_dict(fcst_ds, start_date=start_date)
    else:
        S = datetime.datetime(
            int(start_date[4:8]), ife.strftimeb2int(start_date[0:3]), 1
        )
        return ife.targets_dict(fcst_ds, S)


def get_fcst(fcst_conf, start_date=None, lead_time=None):
    if fcst_conf["method"] == "pycpt":
        fcst_ds, obs = pycpt.get_fcst(fcst_conf)
        start_dates = sorted(
            fcst_ds["S"].dt.strftime(fcst_conf["issue_format"]).values, reverse=True
        )
        if "Li" in fcst_ds.dims :
            target_display = True if (fcst_ds["Li"].size > 1) else False
        else :
            target_display = False
        if start_date is not None:
            fcst_ds = fcst_ds.sel(S=start_date)
        if (("Li" in fcst_ds.dims) and (lead_time is not None)):
            fcst_ds = fcst_ds.sel(Li=int(lead_time))
            obs = obs.where(
                obs["T"].dt.month == fcst_ds.squeeze()["T"].dt.month.values,
                drop=True,
            )
    else:
        S = None if start_date is None else (start_date[0:3] + start_date[4:8])
        fcst_mean_ds, fcst_coeff_ds, fcst_clim_coeff_ds = ife.read_elr(
            Path(fcst_conf["forecast_path"]), fcst_conf["var"], S=S
        )
        fcst_ds = (
            fcst_mean_ds
            .rename({"coeff": "fcst_mean", "prob": "mod"})
            .merge(fcst_coeff_ds)
            .rename({"longitude": "X", "latitude": "Y"})
        )
        obs = fcst_clim_coeff_ds.rename({"longitude": "X", "latitude": "Y"})
        # This is needed because we don't want the map centered in the Pacific
        # and because anyway leaflet (or pingrid?) is confused with X from 0 to 360
        fcst_ds = fcst_ds.assign_coords(X=(((fcst_ds.X + 180) % 360) - 180)).sortby("X")
        obs = obs.assign_coords(X=(((obs.X + 180) % 360) - 180)).sortby("X")
        start_dates = [
            sd.strftime(fcst_conf["issue_format"])
            for sd in ife.get_elr_S(
                Path(fcst_conf["forecast_path"]), fcst_conf["var"]
            )
        ]
        target_display = True
        if lead_time is not None:
            fcst_ds = fcst_ds.sel(lead=int(lead_time))
            obs = obs.sel(lead=int(lead_time))
    return fcst_ds, obs, start_dates, target_display


def errors_handling(config, fcst_ds, obs, lat, lng):
    if config["method"] == "pycpt" :
        fcst_ds_input1 = "deterministic"
        fcst_ds_input2 = "prediction_error_variance"
    else:
        fcst_ds_input1 = "fcst_mean"
        fcst_ds_input2 = "coeff"
    error_msg = None
    try:
        if (
            fcst_ds[fcst_ds_input1] is None
            or fcst_ds[fcst_ds_input2] is None
            or obs is None
        ):
            error_msg="Data missing for this issue and target"
        else:
            fcst_ds = pingrid.sel_snap(fcst_ds, lat, lng)
            obs = pingrid.sel_snap(obs, lat, lng)
            isnan = (
                np.isnan(fcst_ds[fcst_ds_input1]).all()
                or np.isnan(obs).any()
            )
            if config["y_transform"]:
                if fcst_ds["hcst"] is None:
                    error_msg="Data missing for this issue and target"
                else:
                    isnan_yt = np.isnan(fcst_ds["hcst"]).any()
                    isnan = isnan or isnan_yt
            if isnan:
                error_msg="Data missing at this location"
    except KeyError:
        error_msg="Grid box out of data domain"
    return fcst_ds, obs, error_msg


def distribution(quantiles, obs, fcst_ds, config):
    if config["method"] == "pycpt" :
        fcst_ppf, obs_ppf, obs_quant, fcst_pdf, obs_pdf = pycpt.distribution(
            quantiles, obs, fcst_ds, config
        )
        poe = fcst_ppf["percentile"] * -1 + 1
        cdf_fcst_x = fcst_ppf
        cdf_fcst_y = poe
        cdf_obs_x = obs_ppf
        cdf_obs_y = poe
        cdf_obs_emp_x = obs_quant
        cdf_obs_emp_y = poe
        pdf_fcst_x = fcst_ppf
        pdf_fcst_y = fcst_pdf
        pdf_obs_x = obs_ppf
        pdf_obs_y = obs_pdf
    else :
        fcst_cdf, obs_cdf, obs_ppf, fcst_pdf, obs_pdf = ife.ELR_distribution(
            quantiles, obs, fcst_ds
        )
        cdf_fcst_x = obs_ppf
        cdf_fcst_y = fcst_cdf
        cdf_obs_x = obs_ppf
        cdf_obs_y = obs_cdf
        cdf_obs_emp_x = None
        cdf_obs_emp_y = None
        pdf_fcst_x = obs_ppf
        pdf_fcst_y = fcst_pdf
        pdf_obs_x = obs_ppf
        pdf_obs_y = obs_pdf
    return (
        cdf_fcst_x, cdf_fcst_y, cdf_obs_x, cdf_obs_y, cdf_obs_emp_x, cdf_obs_emp_y,
        pdf_fcst_x, pdf_fcst_y, pdf_obs_x, pdf_obs_y,
    )
    

def proba(variable, obs, percentile, config, threshold, fcst_ds):
    if config["method"] == "pycpt" :
        return pycpt.proba(variable, obs, percentile, config, threshold, fcst_ds)
    else:
        if variable == "Percentile":
            threshold = None
        else:
            percentile = None
        if config["variable"] == "Precipitation" :
            clipv = 0
        return ife.proba(
            fcst_ds, obs, percentile=percentile, threshold=threshold, clipv=clipv
        ).rename(X="lon", Y="lat")
    