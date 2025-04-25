from cftime import Datetime360Day as DT360
from dash import html
import dash_bootstrap_components as dbc
import datetime
import io
import numpy as np
import pandas as pd
from collections import OrderedDict
import xarray as xr

import fbfmaproom

def test_year_label_oneyear():
    assert fbfmaproom.year_label(
        DT360(1961, 6, 1),
        1
    ) == "1961"

def test_year_label_straddle():
    assert fbfmaproom.year_label(
        DT360(1960, 12, 16),
        3
    ) == "1960/61"

def test_from_month_since_360Day():
    assert fbfmaproom.from_month_since_360Day(735.5) == DT360(2021, 4, 16)

def test_table_cb():
    table = fbfmaproom.table_cb.__wrapped__(
        issue_month_abbrev = 'feb',
        freq=30,
        mode='0',
        geom_key='ET05',
        pathname='/fbfmaproom/ethiopia',
        severity=0,
        predictand_key="bad-years",
        predictor_keys=['pnep', 'rain', 'ndvi-jan', 'enso_state'],
        season_id='season1',
        include_upcoming='true',
    )

    thead, tbody = table.children
    assert len(thead.children) == 8
    assert len(thead.children[0].children) == 6

    assert thead.children[0].children[0].children == 'Year'
    assert thead.children[0].children[1].children[0].children == 'Forecast prob non-exc v1 (percent)'
    trigger_a = thead.children[1].children[1].children
    assert  isinstance(trigger_a, html.A)
    assert isinstance(trigger_a.children[0], dbc.Button)

    assert thead.children[2].children[0].children[0].children == 'Worthy-action:'
    assert thead.children[3].children[0].children[0].children == 'Act-in-vain:'
    assert thead.children[4].children[0].children[0].children == 'Fail-to-act:'
    assert thead.children[5].children[0].children[0].children == 'Worthy-Inaction:'
    assert thead.children[6].children[0].children[0].children == 'Rate:'
    assert thead.children[7].children[0].children[0].children == 'Threshold:'
    assert thead.children[7].children[1].children == '31.0'

    assert thead.children[0].children[4].children == "ENSO State"
    assert thead.children[2].children[4].children == "9"
    assert thead.children[3].children[4].children == "21"
    assert thead.children[4].children[4].children == "6"
    assert thead.children[5].children[4].children == "3"
    assert thead.children[6].children[4].children == "30.77%"
    assert thead.children[7].children[4].children == "Neutral" # threshold

    assert len(tbody.children) >= 41

    row = tbody.children[-37]
    assert row.children[0].children == '2019'
    assert row.children[0].className == ''
    assert row.children[1].children == '25.4'
    assert row.children[1].className == ''
    assert row.children[2].children == '44.0'
    assert row.children[2].className == ''
    assert row.children[3].children == '0.2218'
    assert row.children[3].className == ''
    assert row.children[4].children == 'El Niño'
    assert row.children[4].className == 'cell-severity-0'
    assert row.children[5].children == ''
    assert row.children[5].className == ''


# overlaps with test_generate_tables, but this one uses synthetic
# data. Merge them?
def test_augment_table_data():
    # year       2022 2021 2020 2019 2018 2017
    # bad_year   T    F    T    F    F    T
    # enso_state           T    F    T    F
    # enso_summ            tp   tn   fp   fn
    # worst_obs  F    F    F    F    F    T
    # obs_summ             fn   tn   tn   tp
    # worst_pnep      F    F    T    F    F
    # pnep_summ       tn   fn   fp   tn   fn
    time = [DT360(y, 1, 16) for y in range(2022, 2016, -1)]
    main_df = pd.DataFrame(
        index=xr.coding.cftimeindex.CFTimeIndex(time),
        data={
            "bad-years": [1, 0, 1, 0, 0, 1],
            "enso_state": [np.nan, np.nan, 3, 1, 3, 2],
            "pnep": [np.nan, 19.606438, 29.270180, 33.800949, 12.312943, 1.],
            "rain": [np.nan, np.nan, 200., 400., 300., 100.],
            "time": time,
        }
    )
    freq = 34
    table_columns = {
        "bad-years": {
            "lower_is_worse": False,
            "type": fbfmaproom.ColType.OBS,
        },
        "pnep": {
            "lower_is_worse": False,
            "type": fbfmaproom.ColType.FORECAST,
        },
        "rain": {
            "lower_is_worse": True,
            "type": fbfmaproom.ColType.OBS,
        },
        "enso_state": {
            "lower_is_worse": False,
            "type": fbfmaproom.ColType.OBS,
        },
    }

    aug, summ, thresholds = fbfmaproom.augment_table_data(main_df, freq, table_columns, "bad-years", final_season=None)

    expected_aug = main_df.copy()
    expected_aug["worst_bad-years"] = [1, 0, 1, 0, 0, 1]
    expected_aug["worst_pnep"] = [np.nan, 0, 0, 1, 0, 0]
    expected_aug["worst_rain"] = [np.nan, np.nan, 0, 0, 0, 1]
    expected_aug["worst_enso_state"] = [np.nan, np.nan, 1, 0, 1, 0]
    pd.testing.assert_frame_equal(expected_aug, aug, check_column_type=True)

    expected_summ = pd.DataFrame(dict(
        # [tp, fp, fn, tn, accuracy]
        pnep=[0, 1, 2, 2, .4],
        rain=[1, 0, 1, 2, .75],
        enso_state=[1, 1, 1, 1, .5],
    ))
    pd.testing.assert_frame_equal(expected_summ, summ)

    assert np.isclose(thresholds["pnep"], 33.8009)
    assert np.isclose(thresholds["rain"], 100)
    assert np.isclose(thresholds["enso_state"], 3)

def test_forecast_tile_url_callback_yesdata():
    url, is_alert, colormap = fbfmaproom.tile_url_callback.__wrapped__(
        2021, 'mar', 30, '/fbfmaproom/ethiopia', 'pnep', 'season1'
    )
    assert url == '/fbfmaproom-tiles/forecast/pnep/{z}/{x}/{y}/ethiopia/season1/2021/2/30'
    assert not is_alert
    assert type(colormap) == list

def test_forecast_tile_url_callback_nodata():
    url, is_alert, colormap = fbfmaproom.tile_url_callback.__wrapped__(
        3333, 'feb', 30, '/fbfmaproom/ethiopia', 'pnep', 'season1'
    )
    assert url == ''
    assert is_alert
    assert type(colormap) == list

def test_forecast_tile():
    with fbfmaproom.SERVER.test_client() as client:
        resp = client.get('/fbfmaproom-tiles/forecast/pnep/6/40/27/ethiopia/season1/2021/2/30')
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"

def test_obs_tile():
    with fbfmaproom.SERVER.test_client() as client:
        resp = client.get('/fbfmaproom-tiles/obs/rain/6/40/27/ethiopia/season1/2021')
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"

def test_vuln_tile():
    with fbfmaproom.SERVER.test_client() as client:
        resp = client.get("/fbfmaproom-tiles/vuln/6/39/30/ethiopia/0/2019")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"

def test_trigger_check_pixel_trigger():
    with fbfmaproom.SERVER.test_client() as client:
        r = client.get(
            "/fbfmaproom/trigger_check?country_key=ethiopia"
            "&variable=pnep"
            "&mode=pixel"
            "&season=season1"
            "&issue_month=1"
            "&season_year=2021"
            "&freq=15"
            "&thresh=10"
            "&bounds=[[6.75, 43.75], [7, 44]]"
        )
    assert r.status_code == 200
    d = r.json
    assert np.isclose(d["value"], 10.7093)
    assert d["triggered"] is True

def test_trigger_check_pixel_notrigger():
    with fbfmaproom.SERVER.test_client() as client:
        r = client.get(
            "/fbfmaproom/trigger_check?country_key=ethiopia"
            "&variable=pnep"
            "&mode=pixel"
            "&season=season1"
            "&issue_month=1"
            "&season_year=2021"
            "&freq=15"
            "&thresh=20"
            "&bounds=[[6.75, 43.75], [7, 44]]"
        )
    assert r.status_code == 200
    d = r.json
    assert np.isclose(d["value"], 10.7093)
    assert d["triggered"] is False

def test_trigger_check_region():
    with fbfmaproom.SERVER.test_client() as client:
        r = client.get(
            "/fbfmaproom/trigger_check?country_key=ethiopia"
            "&variable=pnep"
            "&mode=2"
            "&season=season1"
            "&issue_month=1"
            "&season_year=2021"
            "&freq=15"
            "&thresh=20"
            "&region=ET050501"
        )
    print(r.data)
    assert r.status_code == 200
    d = r.json
    assert np.isclose(d["value"], 9.333)
    assert d["triggered"] is False

def test_trigger_check_straddle():
    "Lead time spans Jan 1"
    with fbfmaproom.SERVER.test_client() as client:
        r = client.get(
            "/fbfmaproom/trigger_check?country_key=malawi"
            "&variable=pnep"
            "&mode=0"
            "&season=season1"
            "&issue_month=10"
            "&season_year=2021"
            "&freq=30.0"
            "&thresh=30.31437"
            "&region=152"
        )
    print(r.data)
    assert r.status_code == 200
    d = r.json
    assert np.isclose(d["value"], 33.10532)
    assert d["triggered"] is True


def test_trigger_check_obs_pixel_trigger():
    with fbfmaproom.SERVER.test_client() as client:
        r = client.get(
            "/fbfmaproom/trigger_check?country_key=ethiopia"
            "&variable=rain"
            "&mode=pixel"
            "&season=season1"
            "&issue_month=1"
            "&season_year=2021"
            "&freq=15"
            "&thresh=90"
            "&bounds=[[6.75, 43.75], [7, 44]]"
        )
    assert r.status_code == 200
    d = r.json
    assert np.isclose(d["value"], 81.154)
    assert d["triggered"] is True

def test_trigger_check_obs_pixel_notrigger():
    with fbfmaproom.SERVER.test_client() as client:
        r = client.get(
            "/fbfmaproom/trigger_check?country_key=ethiopia"
            "&variable=rain"
            "&mode=pixel"
            "&season=season1"
            "&issue_month=1"
            "&season_year=2021"
            "&freq=15"
            "&thresh=20"
            "&bounds=[[6.75, 43.75], [7, 44]]"
        )
    assert r.status_code == 200
    d = r.json
    assert np.isclose(d["value"], 81.154)
    assert d["triggered"] is False

def test_trigger_check_obs_region():
    with fbfmaproom.SERVER.test_client() as client:
        r = client.get(
            "/fbfmaproom/trigger_check?country_key=ethiopia"
            "&variable=pnep"
            "&mode=2"
            "&season=season1"
            "&issue_month=1"
            "&season_year=2021"
            "&freq=15"
            "&thresh=20"
            "&region=ET050501"
        )
    print(r.data)
    assert r.status_code == 200
    d = r.json
    assert np.isclose(d["value"], 9.333)
    assert d["triggered"] is False

def test_trigger_check_forecast_future():
    year = datetime.datetime.now().year + 2
    with fbfmaproom.SERVER.test_client() as client:
        r = client.get(
            "/fbfmaproom/trigger_check?country_key=ethiopia"
            "&variable=pnep"
            "&mode=pixel"
            "&season=season1"
            "&issue_month=1"
            f"&season_year={year}"
            "&freq=15"
            "&thresh=20"
            "&bounds=[[6.75, 43.75], [7, 44]]"
        )
    print(r.data)
    assert r.status_code == 404

def test_trigger_check_obs_future():
    year = datetime.datetime.now().year + 2
    with fbfmaproom.SERVER.test_client() as client:
        r = client.get(
            "/fbfmaproom/trigger_check?country_key=ethiopia"
            "&variable=pnep"
            "&mode=2"
            "&season=season1"
            "&issue_month=1"
            f"&season_year={year}"
            "&freq=15"
            "&thresh=20"
            "&region=ET050501"
        )
    print(r.data)
    assert r.status_code == 404

def test_stats():
    with fbfmaproom.SERVER.test_client() as client:
        resp = client.get('/fbfmaproom-admin/stats')
        print(resp.data)
        assert resp.status_code == 200

def test_update_selected_region_pixel():
    feature_collection, key = fbfmaproom.update_selected_region.__wrapped__(
        [6.875, 43.875],
        'pixel',
        '/fbfmaproom/ethiopia',
    )
    expected = {
        'features': [
            {
                'type': 'Polygon',
                'coordinates': ((
                    (43.75, 6.75),
                    (44.0, 6.75),
                    (44.0, 7.0),
                    (43.75, 7.0),
                    (43.75, 6.75),
                ),)
            }
        ]
    }
    assert feature_collection == expected
    assert key == "[[6.75, 43.75], [7.0, 44.0]]"

def test_update_selected_region_level0():
    feature_collection, key = fbfmaproom.update_selected_region.__wrapped__(
        [6.875, 43.875],
        '0',
        '/fbfmaproom/ethiopia',
    )
    assert len(feature_collection['features'][0]['coordinates'][0][0]) == 1322
    assert key == "ET05"

def test_update_selected_region_level1():
    feature_collection, key = fbfmaproom.update_selected_region.__wrapped__(
        [6.875, 43.875],
        '1',
        '/fbfmaproom/ethiopia',
    )
    assert len(feature_collection['features'][0]['coordinates'][0][0]) == 143
    assert key == "ET0505"

def test_update_popup_pixel():
    content = fbfmaproom.update_popup.__wrapped__(
        '/fbfmaproom/ethiopia',
        [6.875, 43.875],
        'pixel',
    )
    print(content)
    assert isinstance(content, list)
    assert len(content) == 1
    assert isinstance(content[0], html.H3)
    assert content[0].children == '6.87500° N 43.87500° E'

def test_update_popup_level0():
    content = fbfmaproom.update_popup.__wrapped__(
        '/fbfmaproom/ethiopia',
        [6.875, 43.875],
        '0',
    )
    print(content)
    assert isinstance(content, list)
    assert len(content) == 1
    assert isinstance(content[0], html.H3)
    assert content[0].children == 'Somali'

def test_hits_and_misses():
    # year       1960 1961 1962 1963 1964 1965 1966 1967
    # prediction T    F    T    F    F    T
    # truth                T    F    T    F    T    F
    # true_pos             1
    # false_pos                           1
    # false_neg                      1
    # true_neg                  1
    prediction = pd.Series(
        data=[True, False, True, False, False, True],
        index=[DT360(1960 + x, 1, 1) for x in range(6)]
    )
    truth = pd.Series(
        data=[True, False, True, False, True, False],
        index=[DT360(1962 + x, 1, 1) for x in range(6)]
    )
    true_pos, false_pos, false_neg, true_neg, pct = fbfmaproom.hits_and_misses(prediction, truth)
    assert true_pos == 1
    assert false_pos == 1
    assert false_neg == 1
    assert true_neg == 1
    assert pct == .5

def test_format_timedelta_number():
    td = pd.Timedelta(days=3.14159)
    assert fbfmaproom.format_timedelta_days(td) == "3.14"

def test_format_timedelta_nan():
    assert fbfmaproom.format_timedelta_days(pd.NaT) == ""

def test_export_endpoint():
    with fbfmaproom.SERVER.test_client() as client:
        resp = client.get(
            '/fbfmaproom/ethiopia/export'
            '?mode=0'
            '&season=season1'
            '&issue_month0=0'
            '&freq=30'
            '&region=ET05'
            '&predictor=pnep'
            '&predictand=bad-years'
        )
    assert resp.status_code == 200
    d = resp.json

    s = d['skill']
    assert s['act_in_vain'] == 3
    assert s['fail_to_act'] == 6
    assert s['worthy_action'] == 6
    assert s['worthy_inaction'] == 16
    assert np.isclose(s['accuracy'], .70968)

    assert np.isclose(d['threshold'], 29.9870)

    h = d['history']
    assert np.isnan(h[-40]['bad-years'])
    assert np.isnan(h[-40]['worst_bad-years'])
    assert np.isclose(h[-40]['pnep'], 22.9959)
    assert h[-40]['worst_pnep'] == 0
    assert h[-35]['worst_pnep'] == 1

    assert h[-39]['bad-years'] == 1
    assert h[-39]['worst_bad-years'] == 1
    assert h[-38]['bad-years'] == 0
    assert h[-38]['worst_bad-years'] == 0


def test_regions_endpoint():
    with fbfmaproom.SERVER.test_client() as client:
        resp = client.get('/fbfmaproom/regions?country=ethiopia&level=1')
        assert resp.status_code == 200
        d = resp.json
        regions = d['regions']
        assert len(regions) == 11
        assert regions[0]['key'] == 'ET0508'
        assert regions[0]['label'] == 'Afder'


def test_validate_ok():
    contents = (
'''1981,ET05,9
1982,ET05,8
1981,ET0501,7
''')
    errors, notes = fbfmaproom.validate_csv('ethiopia', contents)
    assert len(errors) == 0
    assert len(notes) > 0

def test_validate_ill_formed():
    contents = "<HTML>error</HTML>"
    errors, notes = fbfmaproom.validate_csv('ethiopia', contents)
    assert len(errors) == 1

def test_validate_multiple_errors():
    contents = '1982,XXX,6\n1983,,5'
    errors, notes = fbfmaproom.validate_csv('ethiopia', contents)
    assert len(errors) == 2

def test_validate_unknown_region():
    contents = '1982,XXX,6'
    errors, notes = fbfmaproom.validate_csv('ethiopia', contents)
    assert len(errors) == 1
    assert not errors[0].endswith('...')

def test_validate_many_unknown_regions():
    contents = (
'''1982,XXX,6
1982,XXY,6
1982,XXZ,6
1982,XXA,6
1982,ET05,6
''')
    errors, notes = fbfmaproom.validate_csv('ethiopia', contents)
    assert len(errors) == 1
    assert errors[0].endswith('...')

def test_validate_missing_region():
    contents = '1982,ET05,6\n1982,,6'
    errors, notes = fbfmaproom.validate_csv('ethiopia', contents)
    assert len(errors) == 1

def test_validate_only_missing_region():
    # This case requires special handling because SQL doesn't allow
    # an empty list in WHERE/IN.
    contents = '1982,,6'
    errors, notes = fbfmaproom.validate_csv('ethiopia', contents)
    assert len(errors) == 1

def test_validate_non_int_year():
    contents = 'abc,ET05,6'
    errors, notes = fbfmaproom.validate_csv('ethiopia', contents)
    assert len(errors) == 1

def test_validate_missing_year():
    contents = ',ET05,6'
    errors, notes = fbfmaproom.validate_csv('ethiopia', contents)
    assert len(errors) == 1

def test_validate_non_int_value():
    contents = '1982,ET05,abc'
    errors, notes = fbfmaproom.validate_csv('ethiopia', contents)
    assert len(errors) == 1

def test_validate_missing_value():
    # This is allowed.
    contents = '1981,ET05'
    errors, notes = fbfmaproom.validate_csv('ethiopia', contents)
    assert len(errors) == 0
