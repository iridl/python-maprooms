import flask
import os

from flex_fcst import maproom as flex_fcst
from globals_ import FLASK, GLOBAL_CONFIG
import homepage
from monthly import maproom as monthly
from onset import maproom as onset
import pingrid


@FLASK.route(f"/health")
def health_endpoint():
    return flask.jsonify({'status': 'healthy', 'name': 'python_maproom'})


if __name__ == "__main__":
    if GLOBAL_CONFIG["mode"] != "prod":
        import warnings
        warnings.simplefilter("error")
        debug = True
    else:
        debug = False

    FLASK.run(
        GLOBAL_CONFIG["dev_server_interface"],
        GLOBAL_CONFIG["dev_server_port"],
        debug=debug,
        extra_files=os.environ["CONFIG"].split(":"),
        processes=GLOBAL_CONFIG["dev_processes"],
        threaded=False,
    )
