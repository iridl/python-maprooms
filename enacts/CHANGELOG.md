# ENACTS Maprooms

# Change Log for config files

## Backwards-incompatible changes

### `crop_suitability`

* `layers` are now called `map_text`

* each `map_text` entry has the suffix `_layer` dropped

* `map_text` entries need not `id`

* `target_year` is dropped

* `crop_suitability` config value is now a list of dictionaries rather than a single dictionary.

* `param_defaults["target_season"]` is now a number between 0 and 3 corresponding to "DJF", "MAM", "JJA", "SON"

### `flex_fcst`

* `flex_fcst` config value is now a list of dictionaries rather than a single dictionary.

### `onset`

* `onset` config value is now a list of dictionaries rather than a single dictionary.

## Nota Bene

### all maprooms

* It is no longer necessary to set maproom config keys to `null` in the config file to prevent them from appearing. Only maprooms that are explicitly configured in the config file will be created.
* Country-specific icons should be removed from this repository and added to the country-specific image by the Dockerfile in `python_maproom_mycountry`, e.g.

    ```
    $ ADD metmalawi.png /app/static/
    ```
    