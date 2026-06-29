# ENACTS Maprooms

# Installation and Run Instructions

## Get Shapefiles

* Get the GAUL 2015 levels 0 and 1 from FAO at `https://data.apps.fao.org/catalog/organization/administrative-boundaries-fao?__cf_chl_f_tk=OgUUh7I4P2USfARHuVjb3Rfoqf7aqn2I9Rk18Y390T8-1782743011-1.0.1.1-K6UEdWxnYz2bczx3DjVit2KF0TaXe9b3nV.wnhusfMQ`

## Running the application in a development environment

* Create a development configuration file by copying `config-dev-sample.yaml` to `config-dev.yaml` and editing it as needed, in particular to overwrite the default configuration values set in `config-defaults.yaml` such as data files paths and names. Note that `config-dev.yaml` is in the `.gitignore` file so you won't accidentally commit changes that are specific to your development environment.

* Start the development server using your development config file, e.g.:

    `CONFIG=config-dev.yaml pixi run python app.py`

* Navigate your browser to the URL that is displayed when the server starts, e.g. `http://127.0.0.1:8050/python_maproom/`

* When done using the server stop it with CTRL-C.


