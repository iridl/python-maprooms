import flask
import importlib
import os

from globals_ import FLASK, GLOBAL_CONFIG
import homepage
import pingrid

for name, config in GLOBAL_CONFIG['maprooms'].items():
    if config is not None:
        module = importlib.import_module(name)
        if isinstance(config, list):
            for c in config:
                module.register(FLASK, c)


@FLASK.route(f"{GLOBAL_CONFIG['url_path_prefix']}/health")
def health_endpoint():
    return flask.jsonify({'status': 'healthy', 'name': 'python_maproom'})


if __name__ == "__main__":
    if GLOBAL_CONFIG["mode"] != "prod":
        import warnings
        warnings.simplefilter("error")
        debug = True
    else:
        debug = False

    ssl_context = None
    if "agriculture" in GLOBAL_CONFIG.get("maprooms", {}):
        ssl_cert = GLOBAL_CONFIG.get("ssl_cert")
        ssl_key = GLOBAL_CONFIG.get("ssl_key")
        if ssl_cert and ssl_key:
            ssl_context = (ssl_cert, ssl_key)
    ssl_context = None
    FLASK.run(
        GLOBAL_CONFIG["dev_server_interface"],
        GLOBAL_CONFIG["dev_server_port"],
        debug=debug,
        extra_files=os.environ["CONFIG"].split(":"),
        processes=GLOBAL_CONFIG["dev_processes"],
        threaded=False,
        ssl_context=ssl_context,
        
    )
