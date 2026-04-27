
# FbF Maproom — Developer Documentation

> **Audience:** developers contributing to or maintaining the AA Design tool.  
> This document covers codebase structure, module responsibilities, data flow, and how the key abstractions fit together. For user-facing setup and environment instructions, see `README.md`.

---

## Table of Contents

1. [High-level overview](#1-high-level-overview)
2. [Repository layout](#2-repository-layout)
3. [Module reference](#3-module-reference)
   - [fbfmaproom.py](#fbfmaproompy)
   - [fbflayout.py](#fbflayoutpy)
   - [fbftable.py](#fbftablepy)
   - [fbf-update-data.py](#fbf-update-datapy)
   - [\_\_about\_\_.py](#__about__py)
4. [Configuration system](#4-configuration-system)
5. [Data model](#5-data-model)
6. [Request / callback flow](#6-request--callback-flow)
7. [API endpoints](#7-api-endpoints)
8. [Key algorithms](#8-key-algorithms)
9. [Dependencies](#9-dependencies)
10. [Environment and packaging](#10-environment-and-packaging)
11. [Docker and deployment](#11-docker-and-deployment)
12. [Known limitations and TODOs](#12-known-limitations-and-todos)

---

## 1. High-level overview

The FbF ("Forecast-based Financing") Maproom is a [Dash](https://dash.plotly.com/) web application that lets users evaluate climate forecast skill for anticipatory action (AA) design. It is served by a Flask server and deployed inside a Docker container in production.

The core workflow is:

1. A user selects a **country**, **season**, **forecast dataset**, **observation dataset**, a geographic **region or pixel**, and a **trigger frequency**.
2. The app retrieves historical forecast and observation data (stored as [Zarr](https://zarr.readthedocs.io/) arrays on disk) and computes a **threshold** — the forecast value at or above which the worst `freq`% of years would have been triggered.
3. The app renders an interactive **table** showing year-by-year trigger outcomes and a **map** showing the current season's forecast raster.
4. A "Set trigger" button links out to a companion Gantt tool pre-filled with the calculated threshold. Note: the Gantt integration has been deprecated. The "Set trigger" button needs to be either removed or updated before it can be considered functional.



---

## 2. Repository layout

```
fbfmaproom/
├── __about__.py            # Package metadata (name, version, author)
├── fbfmaproom.py           # Main application: server, Dash app, callbacks, Flask routes
├── fbflayout.py            # Dash component tree (pure layout, no logic)
├── fbftable.py             # HTML table construction helpers
├── fbf-update-data.py      # CLI script: pull data from the IRI Data Library → Zarr
├── fbfmaproom-sample.yaml  # Reference configuration file (never used in prod as-is)
├── pixi.toml               # Dependency spec (pixi / conda-forge)
├── pixi.lock               # Locked dependency tree
├── pyproject.toml          # Pytest configuration
├── Dockerfile              # Production container definition
└── release_container_image # Shell script: build and push Docker image
```

---

## 3. Module reference

### fbfmaproom.py

The entry point and by far the largest file (~1 800 lines). It contains:

#### Server and app setup

```python
SERVER = flask.Flask(__name__)
APP = FbfDash(__name__, server=SERVER, ...)
```

`FbfDash` is a thin subclass of `dash.Dash` that overrides `index()` to return a 404 for any URL path that doesn't correspond to a configured country. This prevents Dash from silently serving the layout for an arbitrary path.

Config is loaded at module import time from the file(s) named in the `CONFIG` environment variable (colon-separated, later files override earlier ones):

```python
CONFIG = pingrid.load_config(os.environ["CONFIG"])
fill_config(CONFIG)
```

`fill_config()` replaces raw `dict` entries under `countries[*].datasets.*` with typed `ObsDataset` / `ForecastDataset` objects so the rest of the code can call methods on them rather than indexing dicts.

---

#### Dataset classes

```
Dataset (base)
├── ObsDataset      — observation data; has lower_is_worse flag
└── ForecastDataset — forecast data; always higher = worse; has is_poe flag
```

Both classes lazily open their Zarr stores via `open_data_array()` and carry metadata needed for the UI (label, units, colormap, value range, number formatter).

The `units` property uses a sentinel `DEFAULT` object to distinguish "not yet loaded" from `None`, so it can read the value from the file the first time it's needed and cache it.

---

#### Data access functions

| Function | What it does |
|---|---|
| `open_data_array(cfg)` | Opens a Zarr store, renames coordinates according to config `var_names`, attaches colormap and scale metadata. |
| `open_forecast(country_key, forecast_key)` | Thin wrapper; sets scale 0–100 (forecasts are probabilities). |
| `open_obs(country_key, obs_key)` | Thin wrapper; converts `timedelta64` arrays to float days. |
| `select_forecast(...)` | Filters forecast array to a single issue month, optionally a single year, and a single percentile level. Handles the PoE vs non-PoE direction convention. |
| `select_obs(...)` | Filters observation array to a target calendar month across all years, optionally a single year. |
| `retrieve_shapes(country_key, level, ...)` | Fetches admin-boundary polygons either from PostGIS (via psycopg2) or from a zipped shapefile (via Fiona), returning a DataFrame with `key`, `label`, and `the_geom` (Shapely geometry) columns. |

---

#### Table generation pipeline

```
fundamental_table_data()
    → For each FORECAST column: calls select_forecast() at the
      requested issue month and freq percentile, renames the
      target_date coordinate to "time".
    → For each OBS column: calls select_obs() across all years
      for the target calendar month.
    → Resolves each dataset to a scalar per year for the selected
      region by calling value_for_geom() (spatial average for
      gridded data, direct key lookup for pre-aggregated data).
    → Merges forecast and obs datasets on the shared "time" axis,
      drops years before start_year, and sorts descending by time.
    → Returns a single xr.Dataset with one variable per table column.
 
augment_table_data()
    → Converts to a pandas DataFrame, calculates percentile ranks,
      identifies worst-year flags, computes thresholds and
      hits/misses summary.
 
generate_tables()
    → Orchestrates the above two functions; called from both the
      Dash callback and the /export endpoint.
```
 
`hits_and_misses()` is a simple confusion-matrix calculation — it takes a boolean "triggered" series and a boolean "bad year" series and returns `[true_pos, false_pos, false_neg, true_neg, accuracy]`, labeled in the UI as worthy-action, act-in-vain, fail-to-act, worthy-inaction, and rate.
 
---

#### Dash callbacks

All callbacks follow the standard Dash pattern. The main ones are:
 
| Callback | Trigger(s) | Purpose |
|---|---|---|
| `initial_setup` | URL pathname change | Populate all dropdowns (season, mode, predictors, etc.) from config; restore state from query string. |
| `forecast_selectors` | Season or map column change | Determine available years and issue months by inspecting the actual forecast Zarr file. |
| `start_year_selector` | Season or pathname change | Populate the start-year filter dropdown. |
| `map_click` | Map click or page load | Update the draggable marker position. |
| `update_selected_region` | Marker position or mode change | Find which admin region (or pixel bounding box) contains the marker; store its key; draw the outline on the map. |
| `update_popup` | Marker position or mode change | Update the marker popup with region name or lat/lon coordinates. |
| `table_cb` | Most control changes | The main callback: run the full table pipeline and render the HTML table + summary. |
| `tile_url_callback` | Year, issue month, freq, map column changes | Constructs the tile URL for the forecast or obs raster layer and updates the colorbar. Also calls `select_forecast` / `select_obs` upfront to detect missing data and show a warning banner if the requested data doesn't exist yet. |
| `borders` | Pathname or mode change | Fetches all admin-boundary polygons for the current country and admin level and sends them to the GeoJSON borders overlay. Returns empty in pixel mode. |
| `validate_upload` | CSV file upload | Decodes the base64 upload, runs `validate_csv()` (checks column structure, year range, and whether all region keys are known to the app), and displays a pass/fail modal with itemised errors and notes. |
| Query-string sync (clientside) | Most control changes | Serialises all current control values into the URL query string so the page state is bookmarkable and survives reload. Runs entirely client-side. |
 
There is also one **clientside callback** (JavaScript) for toggling the map / table panels — purely a CSS class toggle, no server round-trip needed.

---

#### Flask routes (non-Dash)

See [Section 7](#7-api-endpoints) for full details. These are REST endpoints used by the Gantt tool and potentially other consumers.

---

### fbflayout.py

Defines the Dash component tree. Contains **no callbacks, no data access, and no business logic** — just component definitions. If you want to add, remove, or reorganise UI controls, this is where to do it.

Key layout sections:

- `app_layout()` — top-level `html.Div`; assembles control bar, map column, and table column, plus modals and a disclaimer banner.
- `control_layout()` — the horizontal control bar: dropdowns for mode, forecast, issue month, season, year, severity; the frequency slider; the map/table toggle checklist; and the CSV upload button.
- `map_layout()` — a `dash_leaflet.Map` with tile layers (Street and Topo basemaps), an admin-borders GeoJSON overlay, the forecast raster tile layer, a draggable marker with a popup, a scale bar, and a colorbar.
- `table_layout()` — the reference dataset and dataset dropdowns, the "include upcoming" checkbox, the start-year filter, and a loading wrapper around the table container.
- `control(label, tooltip, component)` — a small helper that wraps any control in a `html.Div` with consistent padding and a tooltip.

---

### fbftable.py

Builds the HTML table from DataFrames produced by the main pipeline. Keeps rendering logic separate from the data logic in `fbfmaproom.py`.

| Function | Purpose |
|---|---|
| `gen_table(tcs, dfs, data, thresholds, severity, final_season)` | Entry point. Produces a complete `html.Table` with header and body. |
| `gen_head(tcs, dfs)` | Builds `<thead>`: one row for column names (with tooltips), then one row per row in the summary DataFrame (`dfs`). |
| `gen_body(tcs, data, thresholds, severity, final_season)` | Builds `<tbody>`: one `<tr>` per year in `data`. |
| `cell_class(col_name, row, severity, thresh, lower_is_worse, final_season)` | Returns a CSS class name based on whether the cell is a triggered year, an excluded (upcoming) year, both, or neither. Four possible classes: `''`, `'cell-excluded'`, `'cell-severity-{0,1,2}'`, `'cell-excluded-severity-{0,1,2}'`. |

`gen_select_header()` renders a `<select>` element in the table header — currently used for column-level controls embedded in the header row.

---

### fbf-update-data.py

A standalone CLI script for refreshing datasets from the IRI Data Library. It is not imported by the web application.

**Usage:**
```bash
CONFIG=fbfmaproom-sample.yaml:config-local.yaml pixi run python fbf-update-data.py ethiopia/rain-mam
```

**What it does:**
1. Reads `CONFIG` to find all datasets that have a `url` field (i.e. are sourced from the Data Library).
2. For each requested dataset (or all of them if none are specified on the command line), downloads the data as NetCDF using `curl`, then converts to Zarr using xarray.
3. If a dataset has multiple `url_args` slices, it downloads each slice separately and concatenates them along the `T` axis before writing.
4. Optionally accepts a `--cookiefile` for authenticated Data Library downloads.
5. Uses `pingrid.open_dataset()` which handles coordinate renaming and calendar normalisation.

Datasets not sourced from the Data Library (e.g. PyCPT forecasts) are converted by separate scripts in `data-conversion-scripts/` and are not covered by this script.

---

### \_\_about\_\_.py

Simple metadata file. Provides `name`, `author`, `email`, and `version`. Imported by `fbfmaproom.py` to expose the version string via the `/fbfmaproom-admin/stats` endpoint.

---

## 4. Configuration system

Configuration is stored in one or more YAML files, merged at startup. The expected pattern for local development is:

```
CONFIG=fbfmaproom-sample.yaml:config-local.yaml
```

`config-local.yaml` overrides only the fields that differ in your environment (typically `db.password` and `dev_server_port`). In production, a separate secrets file is injected by the deployment process.

### Top-level keys

| Key | Type | Purpose |
|---|---|---|
| `mode` | `debug` / `devel` / `prod` | Controls debug flags and warning behaviour. |
| `core_path` | string | URL prefix for the Dash app (e.g. `/fbfmaproom`). |
| `tile_path` | string | URL prefix for raster tile endpoints. |
| `admin_path` | string | URL prefix for admin/debug endpoints. |
| `data_root` | path | Root directory for all Zarr datasets. |
| `custom_asset_path` | path | Directory containing per-country logo images. |
| `db` | dict | psycopg2 connection kwargs (`host`, `port`, `dbname`, `user`, `password`). |
| `gantt_url` | string | Base URL of the external Gantt tool; the export endpoint appends query parameters to this. |
| `countries` | dict | Per-country configuration (see below). |

### Per-country keys

| Key | Purpose |
|---|---|
| `logo` | Filename of the country logo PNG in `custom_asset_path`. |
| `center` | `[lon, lat]` — initial map centre. |
| `zoom` | Initial map zoom level. |
| `marker` | `[lon, lat]` — initial marker position. |
| `resolution` | `[dx, dy]` in degrees — pixel size for pixel mode. |
| `shapes` | List of admin levels. Each entry has either a `sql` key (PostGIS query) or `file`/`layer`/`key_field`/`label_field` keys (shapefile). |
| `seasons` | Dict of season configs: `label`, `target_month` (0-based), `length` (months), `issue_months` (list of 0-based month indices), `start_year`. |
| `datasets.defaults` | Default `predictors` (list) and `predictand` (single key). |
| `datasets.observations` | Dict of observation dataset configs. |
| `datasets.forecasts` | Dict of forecast dataset configs. |

### Dataset config keys

| Key | Required for | Purpose |
|---|---|---|
| `label` | all | Display name in dropdowns. |
| `description` | all | Tooltip text. |
| `path` | all | Path relative to `data_root`, including `.zarr` suffix. |
| `var_names` | all | Maps standard names (`value`, `lat`, `lon`, `time`) to the names used in the Zarr file. |
| `colormap` | all | Key into `pingrid.CMAPS`. |
| `format` | all | One of `number0`–`number4`, `timedelta_days`, `bad`, `enso`. |
| `range` | all | `[min, max]` for colorbar scaling. |
| `url` | obs only | Data Library URL (used by `fbf-update-data.py`). |
| `url_args` | obs only | List of dicts for parameterising `url` (one download per entry). |
| `lower_is_worse` | obs only | `true` if low values are the bad outcome (e.g. rainfall). |
| `is_poe` | forecast only | `true` if the forecast is expressed as probability of exceedance. |

---

## 5. Data model

All gridded datasets are stored as [Zarr](https://zarr.readthedocs.io/) arrays on disk. The standard coordinate names after renaming are:

- `time` — cftime `Datetime360Day` values (the IRI Data Library uses a 360-day calendar throughout).
- `lat` / `lon` — spatial coordinates, present for gridded data.
- `issue` — for forecast data, the date the forecast was issued.
- `lead` — lead time in months, present in some forecast formats.
- `pct` — percentile, present in probabilistic forecasts.
- `geom_key` — for non-gridded data indexed by admin region key.

Admin boundaries are stored either in PostGIS (queried via psycopg2) or in zipped shapefiles on disk (read via Fiona). The `the_geom` column is returned as a Shapely geometry object.

---

## 6. Request / callback flow

```
Browser
  │
  ├─ Page load → initial_setup callback
  │     Reads CONFIG for the country in the URL path.
  │     Populates all dropdowns; restores state from query string.
  │
  ├─ User changes any control
  │     → One or more callbacks fire (see table in §3).
  │     → table_cb is the most expensive: opens Zarr files,
  │        averages over the selected region, ranks years,
  │        and re-renders the entire table.
  │
  ├─ User clicks marker on map
  │     → map_click updates marker position.
  │     → update_selected_region finds the admin region or pixel
  │        bounding box and stores the geometry key.
  │     → table_cb and raster callbacks re-fire.
  │
  └─ User clicks "Set trigger"
        → External link to the Gantt tool, with threshold and
          region parameters encoded in the query string.
          No server call from the Maproom.
```

---

## 7. API endpoints

These are Flask routes, separate from the Dash callback system. They return JSON (or YAML for the stats endpoint) and are used in two contexts: by external scripts maintained by the FIST team (e.g. the [AAF](https://bitbucket.org/iri-fist/workspace/projects/AAF) project), and historically by the Gantt tool integration, which has since been deprecated.
### `GET /fbfmaproom-admin/stats`
Health check. Returns version string and process/thread stats (pid, active thread count, per-thread name and daemon status) as YAML. Note: uses YAML rather than JSON — the comment in the code explicitly flags this as a pattern not to imitate.
 
### `GET /fbfmaproom-tiles/forecast/<forecast_key>/<z>/<x>/<y>/<country_key>/<season_id>/<target_year>/<issue_month0>/<freq>`
Serves a 256×256 PNG map tile for a forecast dataset at the given zoom/tile coordinates. Clips the raster to the national boundary unless `clip: false` is set in the country config.
 
### `GET /fbfmaproom-tiles/obs/<obs_key>/<z>/<x>/<y>/<country_key>/<season_id>/<target_year>`
Serves a 256×256 PNG map tile for an observation dataset. Always clips to the national boundary.
 
### `GET /fbfmaproom/trigger_check`
Checks whether a forecast or observation value for a given region and year meets a threshold.
 
Query params: `variable`, `country_key`, `mode`, `season`, `issue_month0`, `season_year`, `freq`, `thresh`. In pixel mode, also `bounds`; in admin mode, also `region`.
 
Returns: `{"value": <float>, "triggered": <bool>}`.
 
### `GET /fbfmaproom/<country_key>/export`
Returns the full hits/misses summary and year-by-year history for a given predictor/predictand pair. Used by the Gantt tool to display historical trigger performance.
 
Query params: `mode`, `season`, `issue_month` (or `issue_month0`), `freq`, `region`, `predictor`, `predictand`, `include_upcoming`, `start_year`.
 
Returns a JSON object with `skill` (worthy_action, act_in_vain, fail_to_act, worthy_inaction, accuracy), `history` (list of year records), and `threshold`.
 
### `GET /fbfmaproom/regions/geojson`
Returns admin-level polygons as a GeoJSON FeatureCollection.
 
Query params: `country`, `level`, `region` (optional, filter to one region).
 
### `GET /fbfmaproom/regions`
Returns admin region keys and labels as a JSON list (no geometry).
 
Query params: `country`, `level`.
 
### `GET /fbfmaproom/custom/<path>`
Serves static assets (logos etc.) from `custom_asset_path`.

---

## 8. Key algorithms

### Threshold calculation

For a given `freq` (e.g. 30 for "30% worst years"):

1. Drop NaN values and rank the historical series by percentile.
2. A year is flagged as "worst" if its rank falls in the bottom `freq`% (for ascending variables like rainfall) or top `freq`% (for descending variables like forecast probability).
3. The threshold is the maximum worst-year value (for ascending) or the minimum worst-year value (for descending).

Special case: if the dataset has only two unique values (legacy boolean bad-years flag), the worst value is used directly as the threshold regardless of `freq`.

### Forecast percentile selection

Forecasts can be expressed either as probability of exceedance (PoE) or probability of non-exceedance. The `is_poe` flag determines which direction "bad" is:

- `is_poe = True`: higher probability = worse; select the `(100 - freq)`th percentile.
- `is_poe = False`: higher probability = better; select the `freq`th percentile.

### Spatial averaging

When a dataset has `lon`/`lat` coordinates, the value for a region is computed by averaging all grid cells whose centres fall within (or touch) the region polygon, using `pingrid.average_over(..., all_touched=True)`.

When a dataset has a `geom_key` coordinate (pre-aggregated by region), the value is selected directly by key. Cross-region interpolation is noted as not yet implemented.

### Season year labelling

A season that starts in one year and ends in another (e.g. Dec–Feb) is labelled `YYYY/YY` (e.g. `2023/24`). Single-year seasons show just the year.

---

## 9. Dependencies

Core dependencies and the reason each is pinned or notable:

| Package | Notes |
|---|---|
| `dash = 2.3.1.*` | Pinned because a later version removes `__wrapped__` from callbacks, which some tests rely on. |
| `pandas = 1.3.*` | Newer versions emit warnings when passed a psycopg2 connection directly. Upgrade requires switching to SQLAlchemy. |
| `numpy = 1.24.2` | Pinned to suppress a warning caused by the pinned pandas version. |
| `werkzeug < 2.1` | Werkzeug 2.1 emits warnings at dev-server startup. |
| `python = 3.9.*` | Rasterio build fails with Python 3.10 (upstream issue). |
| `gunicorn` | Linux-only; not packaged for Windows. Production WSGI server. |
| `dash-leaflet-iri` | IRI fork of dash-leaflet; from the `iridl` conda channel. |
| `cftime` | Handles the 360-day calendar used throughout the IRI Data Library. |
| `fiona` | Reads zipped shapefiles for countries whose boundaries are stored as files rather than in PostGIS. |

---

## 10. Environment and packaging

Dependencies are managed with [pixi](https://pixi.sh), which wraps conda-forge. The key commands:

```bash
# Install all dependencies (creates .pixi/ environment)
pixi install

# Add a new dependency
pixi add <package>
pixi lock

# Run the dev server
CONFIG=fbfmaproom-sample.yaml:config-local.yaml pixi run python fbfmaproom.py

# Run tests
pixi run pytest
```

The `pixi.toml` defines two environments: `default` (includes dev dependencies like pytest) and `prod` (dependencies only). Both use the same solve group so they share a locked set of packages.

---

## 11. Docker and deployment

The `Dockerfile` builds a production image. The `release_container_image` script builds and pushes it to the registry. In production, gunicorn (not the Dash dev server) is used as the WSGI server.

The `CONFIG` environment variable must be set in the container to point to the production config file(s). Secrets (db password, etc.) are typically injected at runtime via a separate config file or environment variable, not baked into the image.

---

## 12. Known limitations and TODOs

These are noted in the code and worth tracking:

- **Cross-region interpolation** (`value_for_geom`): if a dataset is indexed by geom_key but the selected region is not present, the code raises a generic exception (“Not implemented”). Area-weighted intersection is noted as a future improvement.
- **SQLAlchemy migration**: the direct psycopg2 connection passed to `pd.read_sql` might trigger deprecation warnings in newer pandas. Switching to a SQLAlchemy engine would allow pandas to be unpinned.
- **Colormap / scale attributes in Zarr**: some datasets already embed colormap and scale metadata as Zarr attributes. Currently these are ignored in favour of config-file values. Using them would simplify the config schema.
- **`ImportForeignTable` vs the current IMPORT approach**: noted in the README — the current approach is preferred so table definitions don't have to be hand-specified.
- **Python 3.9 pin**: driven by a rasterio build issue. Worth revisiting when a compatible rasterio release is available.
