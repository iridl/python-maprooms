import pandas as pd


def target_range_format(leads_value, start_date, period_length, time_units):
    """ Formatting target range using leads and starts, and target range period length.

    Parameters
    ----------
    leads_value : int
        Application's integer representation of lead time.
    start_date : Timestamp
        Start date for the forecast.
    period_length : int
       Length of forecast target range period.
    time_units: str
       corresponds to the temporal parameter in which `leads_value`
       and `period_length` are expressed in.
    Returns
    -------
    date_range : str
        String of target date range.
    Notes
    -----
    If the providers representation of lead time is an integer value, convert to str for input.
    The function will output the most concise version of the date range depending on if years and months are equal.
    """
    target_start = start_date + pd.offsets.DateOffset(**{time_units:leads_value}) 
    target_end = target_start + pd.offsets.DateOffset(**{time_units:period_length-1})
    return target_range_formatting(target_start, target_end, time_units)


def target_range_formatting(target_start, target_end, time_units):
    target_start = pd.Timestamp(target_start)
    target_end = pd.Timestamp(target_end)
    if target_start.year == target_end.year:
        if target_start.month == target_end.month:
            target_start_str = target_start.strftime("%-d")
        else:
            if time_units == "days":
                target_start_str = (target_start).strftime("%-d %b")
            else:
                target_start_str = (target_start).strftime("%b")
    else:
        if time_units == "days":
            target_start_str = (target_start).strftime("%-d %b %Y")
        else:
            target_start_str = (target_start).strftime("%b %Y")
    if time_units == "days":
        target_end_str = target_end.strftime("%-d %b %Y")
    else:
        target_end_str = target_end.strftime("%b %Y")
    date_range = f"{target_start_str}-{target_end_str}"
    return date_range

