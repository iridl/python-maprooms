# ENACTS Maprooms

# Installation and Run Instructions

## Creating a conda environment with this project's dependencies

* see enacts' README

## Running the application in a development environment

* Activate the environment

    `conda activate enactsmaproom`

* Create a development configuration file by copying `config-dev-sample.yaml` to `config-dev.yaml` and editing it as needed. Note that `config-dev.yaml` is in the `.gitignore` file so you won't accidentally commit changes that are specific to your development environment.

* Start the development server using your development config file, e.g.:

    `CONFIG=config-dev.yaml python app.py`

* Navigate your browser to the URL that is displayed when the server starts, e.g. `http://127.0.0.1:8050/python_maproom/`

* When done using the server stop it with CTRL-C.

# Development Overview

see enacts' README

## Adding or removing dependencies

see enacts' README

## Building the documentation

see enacts' README

# Docker Build Instructions

TBD

# Support

* `help@iri.columbia.edu`
