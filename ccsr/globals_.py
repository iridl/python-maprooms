import flask
import pingrid
import os

defaultconfig = pingrid.load_config("config-defaults.yaml")
appconfig = pingrid.load_config(os.environ["CONFIG"])
GLOBAL_CONFIG = pingrid.deep_merge(
    {k: v for k, v in defaultconfig.items() if k != "maprooms"},
    {k: v for k, v in appconfig.items() if k != "maprooms"},
)
GLOBAL_CONFIG['maprooms'] = {}
for k, v in appconfig['maprooms'].items():
    if isinstance(v, list):
        GLOBAL_CONFIG['maprooms'][k] = [pingrid.deep_merge(defaultconfig['maprooms'][k], v[i]) for i in range(len(v))]
    elif v is not None:
        GLOBAL_CONFIG['maprooms'][k] = pingrid.deep_merge(defaultconfig['maprooms'][k], v)


FLASK = flask.Flask(
    "ccsrmaprooms",
    static_url_path=f'{GLOBAL_CONFIG["url_path_prefix"]}/static',
)
