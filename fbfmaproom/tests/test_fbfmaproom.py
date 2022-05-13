from cftime import Datetime360Day as DT360
from dash import html
import io
import numpy as np
import pandas as pd

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

def test_generate_tables():
    main_df, summary_df, thresholds = fbfmaproom.generate_tables(
        country_key='ethiopia',
        obs_dataset_key='rain',
        season_config={
            'label': 'MAM',
            'target_month': 3.5,
            'length': 3.0,
            'issue_months': [11, 0, 1],
            'year_range': [1983, 2021]
        },
        table_columns=[
            {'id': 'year_label', 'name': 'Year'},
            {'id': 'enso_state', 'name': 'ENSO State'},
            {'id': 'forecast', 'name': 'Forecast, %'},
            {'id': 'obs_rank', 'name': 'Rain Rank'},
            {'id': 'bad_year', 'name': 'Reported Bad Years'}
        ],
        issue_month0=1,
        freq=30,
        mode='0',
        geom_key='ET05',
        severity=0,
    )

    # for c in main_df.columns:
    #     print(f'{c}={list(main_df[c].values)}')

    expected_main = pd.DataFrame.from_dict(dict(
        time=[DT360(year, 4, 16) for year in range(2021, 1982, -1)],
        year_label=[
            '2021', '2020', '2019', '2018', '2017', '2016', '2015', '2014',
            '2013', '2012', '2011', '2010', '2009', '2008', '2007', '2006',
            '2005', '2004', '2003', '2002', '2001', '2000', '1999', '1998',
            '1997', '1996', '1995', '1994', '1993', '1992', '1991', '1990',
            '1989', '1988', '1987', '1986', '1985', '1984', '1983'
        ],
        enso_state=[
            'La Niña', 'Neutral', 'El Niño', 'La Niña', 'Neutral', 'El Niño',
            'El Niño', 'Neutral', 'Neutral', 'La Niña', 'La Niña', 'Neutral',
            'Neutral', 'La Niña', 'Neutral', 'Neutral', 'Neutral', 'Neutral',
            'Neutral', 'Neutral', 'Neutral', 'La Niña', 'La Niña', 'El Niño',
            'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'El Niño',
            'Neutral', 'Neutral', 'La Niña', 'Neutral', 'El Niño', 'Neutral',
            'La Niña', 'Neutral', 'El Niño'
        ],
        forecast=[
            '34.04', '26.84', '34.28', '32.35', '36.43', '31.38', '32.21',
            '33.35', '38.26', '38.26', '37.05', '30.61', '36.07', '41.86',
            '34.10', '42.46', '36.14', '36.93', '35.87', '31.45', '35.50',
            '40.70', '40.95', '28.68', '35.04', '33.82', '35.92', '37.54',
            '29.53', '39.90', '27.96', '28.82', '34.68', '35.84', '31.13',
            '35.42', '38.82', '38.87', '36.09'
        ],
        obs_rank=[
            24., 33., 12., 38.,  8., 32., 16., 13., 35., 15.,  3., 37.,  5.,
            4., 19., 21., 20.,  9., 14., 10., 11.,  6.,  7., 18., 22., 30.,
            31., 25., 26.,  2., 28., 27., 34., 23., 39., 29., 36.,  1., 17.
        ],
        bad_year=[
            '', '', '', '', 'Bad', 'Bad', '', '', '', '', 'Bad',
            '', 'Bad', 'Bad', '', '', 'Bad', '', '', '', '', 'Bad',
            'Bad', '', '', '', '', '', '', 'Bad', '', '', '',
            '', '', '', '', 'Bad', ''
        ],
        forecast_numeric=[
            34.036846, 26.84371, 34.275295, 32.347477, 36.428505, 31.376135, 32.212322,
            33.347603, 38.258884, 38.258804, 37.052757, 30.61248,  36.068275, 41.856205,
            34.0954,   42.46017,  36.144985, 36.93086,  35.872562, 31.448317, 35.4979,
            40.69789,  40.95002,  28.682444, 35.03906,  33.82416,  35.923187, 37.54364,
            29.528564, 39.904465, 27.96153,  28.823236, 34.679913, 35.844246, 31.127577,
            35.4181,   38.823475, 38.869564, 36.086605
        ],
    )).set_index("time")
    pd.testing.assert_frame_equal(main_df, expected_main, check_index_type=False)

    # for c in summary_df.columns:
    #     print(f'{c}={list(summary_df[c].values)}')
    expected_summary = pd.DataFrame.from_dict(dict(
        year_label=['Worthy-action:', 'Act-in-vain:', 'Fail-to-act:',
                    'Worthy-Inaction:', 'Rate:', 'Year'],
        enso_state=[2, 5, 8, 24, '66.67%', 'ENSO State'],
        forecast=[6, 5, 4, 24, '76.92%', 'Forecast, %'],
        obs_rank=[8, 3, 2, 26, '87.18%', 'Rain Rank'],
        bad_year=[None, None, None, None, None, 'Reported Bad Years'],
    ))
    pd.testing.assert_frame_equal(
        summary_df.set_index("year_label"),
        expected_summary.set_index("year_label")
    )

    np.testing.assert_equal(
        (main_df["obs_rank"] <= thresholds["obs"]).values,
        [
            False, False, False, False, True, False, False, False,
            False, False, True, False, True, True, False, False, False,
            True, False, True, True, True, True, False, False, False,
            False, False, False, True, False, False, False, False, False,
            False, False, True, False
        ]
    )
    print(main_df.columns)
    np.testing.assert_equal(
        (main_df["forecast_numeric"] >= thresholds["pnep"]).values,
        [
            False, False, False, False, False, False, False, False,
            True, True, True, False, False, True, False, True, False,
            False, False, False, False, True, True, False, False,
            False, False, True, False, True, False, False, False,
            False, False, False, True, True, False
        ],
    )

    assert np.isclose(thresholds["pnep"], 37.052727)

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
    main_df = pd.DataFrame(
        index=[DT360(y, 1, 16) for y in range(2022, 2016, -1)],
        data={
            "bad_year": [True, False, True, False, False, True],
            "enso_state": [np.nan, np.nan, "El Niño", "La Niña", "El Niño", "Neutral"],
            "obs": [np.nan, np.nan, 200., 400., 300., 100.],
            "pnep": [np.nan, 19.606438, 29.270180, 33.800949, 12.312943, 1.],
        }
    )
    freq = 34
    aug, summ, thresholds = fbfmaproom.augment_table_data(main_df, freq)

    expected_aug = main_df.copy()
    expected_aug["obs_rank"] = [np.nan, np.nan, 2, 4, 3, 1]
    pd.testing.assert_frame_equal(expected_aug, aug, check_column_type=True)

    expected_summ = pd.DataFrame(dict(
        # [tp, fp, fn, tn, accuracy]
        enso_state=[1, 1, 1, 1, "50.00%"],
        forecast=[0, 1, 2, 2, "40.00%"],
        obs_rank=[1, 0, 1, 2, "75.00%"],
    ))
    pd.testing.assert_frame_equal(expected_summ, summ)

    np.testing.assert_equal(
        (aug["obs_rank"] <= thresholds["obs"]).values,
        [False, False, False, False, False, True]
    )
    np.testing.assert_equal(
        (aug["pnep"] >= thresholds["pnep"]).values,
        [False, False, False, True, False, False]
    )
    assert np.isclose(thresholds["pnep"], 33.800949)


def test_pnep_tile_url_callback_yesdata():
    url, is_alert = fbfmaproom.pnep_tile_url_callback.__wrapped__(
        2021, 2, 30, '/fbfmaproom/ethiopia', 'season1'
    )
    assert url == '/fbfmaproom-tiles/pnep/{z}/{x}/{y}/ethiopia/season1/2021/2/30'
    assert not is_alert

def test_pnep_tile_url_callback_nodata():
    url, is_alert = fbfmaproom.pnep_tile_url_callback.__wrapped__(
        3333, 2, 30, '/fbfmaproom/ethiopia', 'season1'
    )
    assert url == ''
    assert is_alert

def test_pnep_tiles():
    with fbfmaproom.SERVER.test_client() as client:
        resp = client.get('/fbfmaproom-tiles/pnep/6/40/27/ethiopia/season1/2021/2/30')
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"

def test_vuln_tiles():
    with fbfmaproom.SERVER.test_client() as client:
        resp = client.get("/fbfmaproom-tiles/vuln/6/39/30/ethiopia/0/2019")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"

def test_pnep_percentile_pixel_trigger():
    with fbfmaproom.SERVER.test_client() as client:
        r = client.get(
            "/fbfmaproom/pnep_percentile?country_key=ethiopia"
            "&mode=pixel"
            "&season=season1"
            "&issue_month=1"
            "&season_year=2021"
            "&freq=15"
            "&prob_thresh=10"
            "&bounds=[[6.75, 43.75], [7, 44]]"
        )
    assert r.status_code == 200
    d = r.json
    assert np.isclose(d["probability"], 13.415)
    assert d["triggered"] is True

def test_pnep_percentile_pixel_notrigger():
    with fbfmaproom.SERVER.test_client() as client:
        r = client.get(
            "/fbfmaproom/pnep_percentile?country_key=ethiopia"
            "&mode=pixel"
            "&season=season1"
            "&issue_month=1"
            "&season_year=2021"
            "&freq=15"
            "&prob_thresh=20"
            "&bounds=[[6.75, 43.75], [7, 44]]"
        )
    assert r.status_code == 200
    d = r.json
    assert np.isclose(d["probability"], 13.415)
    assert d["triggered"] is False

def test_pnep_percentile_region():
    with fbfmaproom.SERVER.test_client() as client:
        r = client.get(
            "/fbfmaproom/pnep_percentile?country_key=ethiopia"
            "&mode=2"
            "&season=season1"
            "&issue_month=1"
            "&season_year=2021"
            "&freq=15"
            "&prob_thresh=20"
            "&region=(ET05,ET0505,ET050501)"
        )
    print(r.data)
    assert r.status_code == 200
    d = r.json
    assert np.isclose(d["probability"], 14.6804)
    assert d["triggered"] is False

def test_pnep_percentile_straddle():
    "Lead time spans Jan 1"
    with fbfmaproom.SERVER.test_client() as client:
        r = client.get(
            "/fbfmaproom/pnep_percentile?country_key=malawi"
            "&mode=0"
            "&season=season1"
            "&issue_month=10"
            "&season_year=2021"
            "&freq=30.0"
            "&prob_thresh=30.31437"
            "&region=152"
        )
    print(r.data)
    assert r.status_code == 200
    d = r.json
    assert np.isclose(d["probability"], 33.10532)
    assert d["triggered"] is True


def test_download_table_all_freq():
    with fbfmaproom.SERVER.test_client() as client:
        resp = client.get(
            '/fbfmaproom/download_table?country_key=ethiopia'
            '&obs_dataset_key=rain'
            '&season_id=season1'
            '&issue_month=jan'
            '&mode=0'
            '&geom_key=ET05'
        )
    #print(resp.data)
    assert resp.status_code == 200
    assert resp.mimetype == "text/csv"
    csv_file = io.StringIO(resp.get_data(as_text=True))
    df = pd.read_csv(csv_file)
    print(df.columns)
    assert list(df.columns) == [
       'time', 'bad_year', 'obs', 'enso_state', 'pnep_05', 'pnep_10',
       'pnep_15', 'pnep_20', 'pnep_25', 'pnep_30', 'pnep_35', 'pnep_40',
       'pnep_45', 'pnep_50', 'pnep_55', 'pnep_60', 'pnep_65', 'pnep_70',
       'pnep_75', 'pnep_80', 'pnep_85', 'pnep_90', 'pnep_95'
    ]
    onerow = df[df["time"] == "2019-04-16"]
    assert len(onerow) == 1
    assert onerow["bad_year"].values[0] == 0.0
    assert np.isclose(onerow["obs"].values[0], 3902.611)
    assert np.isclose(onerow["pnep_30"].values[0], 33.700)
    assert onerow["enso_state"].values[0] == "El Niño"

def test_download_table_one_freq():
    with fbfmaproom.SERVER.test_client() as client:
        resp = client.get(
            '/fbfmaproom/download_table?country_key=ethiopia'
            '&obs_dataset_key=rain'
            '&season_id=season1'
            '&issue_month=jan'
            '&mode=0'
            '&geom_key=ET05'
            '&freq=30'
        )
    assert resp.status_code == 200
    assert resp.mimetype == "text/csv"
    csv_file = io.StringIO(resp.get_data(as_text=True))
    df = pd.read_csv(csv_file)
    assert list(df.columns) == [
        'time', 'bad_year', 'obs', 'enso_state', 'worst_pnep', 'pnep_30',
    ]
    onerow = df[df["time"] == "2019-04-16"]
    assert len(onerow) == 1
    assert onerow["bad_year"].values[0] == 0.0
    assert np.isclose(onerow["obs"].values[0], 3902.611)
    assert np.isclose(onerow["pnep_30"].values[0], 33.700)
    assert onerow["enso_state"].values[0] == "El Niño"
    assert onerow["worst_pnep"].values[0] == 0


def test_stats():
    with fbfmaproom.SERVER.test_client() as client:
        resp = client.get('/fbfmaproom-admin/stats')
        print(resp.data)
        assert resp.status_code == 200

def test_update_selected_region_pixel():
    positions, key = fbfmaproom.update_selected_region.__wrapped__(
        '/fbfmaproom/ethiopia',
        [6.875, 43.875],
        'pixel'
    )
    assert positions == [[[6.75, 43.75], [7.0, 43.75], [7.0, 44.0], [6.75, 44.0]]]
    assert key == "[[6.75, 43.75], [7.0, 44.0]]"

def test_update_selected_region_level0():
    positions, key = fbfmaproom.update_selected_region.__wrapped__(
        '/fbfmaproom/ethiopia',
        [6.875, 43.875],
        '0'
    )
    assert len(positions[0]) == 1323
    assert key == "ET05"

def test_update_selected_region_level1():
    positions, key = fbfmaproom.update_selected_region.__wrapped__(
        '/fbfmaproom/ethiopia',
        [6.875, 43.875],
        '1'
    )
    assert len(positions[0]) == 143
    assert key == "(ET05,ET0505)"

def test_update_popup_pixel():
    title, content = fbfmaproom.update_popup.__wrapped__(
        '/fbfmaproom/ethiopia',
        [6.875, 43.875],
        'pixel',
        2017,
    )
    print(repr(content))
    assert isinstance(title, html.H3)
    assert title.children == '6.87500° N 43.87500° E'
    assert content.children == []

def test_update_popup_level0():
    title, content = fbfmaproom.update_popup.__wrapped__(
        '/fbfmaproom/ethiopia',
        [6.875, 43.875],
        '0',
        2017,
    )
    assert isinstance(title, html.H3)
    assert title.children == 'Somali'
    assert len(content.children) == 12
    assert content.children[0].children == 'Vulnerability: '
    assert content.children[1].strip() == '31'

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
    assert pct == "50.00%"
