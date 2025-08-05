# FbF Maproom

## Setting up a development environment

For now, these instructions are specific to developers working on an IRI server.

### Log into the server, with port forwarding enabled

Choose port on which to run your development server. You must choose one that isn't in use by another developer on the same server. Log into the server (mako in the example) and forward your chosen port (xxxx in the example). Your chosen port should be a number greater than 1000. 

An easy way to ensure that you are using a unique port number is to use your Unix User ID. You can determine this ID by logging into the server and running the command 'id -u'. 

```
ssh -L xxxx:localhost:xxxx mako.iri.columbia.edu
```

### On the server, create an ssh key to use with GitHub

```
ssh-keygen -f ~/.ssh/id_rsa
```
Provide a strong password when prompted. Then
```
cat ~/.ssh/id_rsa.pub
```
and copy the output of that command for use in the next one. It should start with `ssh-rsa`.

### Configure GitHub to use that ssh key
In a browser, visit [https://github.com/settings/ssh/new]. For the title put e.g. "IRI development server," and in the Key box paste the output from the previous comand. Then click the green "Add SSH key" button.

### Clone this repository and enter the fbfmaproom directory

```
git clone git@github.com:iridl/python-maprooms
cd python-maprooms/fbfmaproom
```

### Enable conda

```
module load python/miniconda3
```

### Create a local configuration file

Create a file called `config-local.yaml` with the following contents, modified as noted in the comments.
```
db:
    password: itsasecret # get the real db password from another developer
dev_server_port: xxxx # the port you chose earlier
```

## Running the application in your development environment
### Activate the conda environment (pre-installed on the development server)
```
conda activate fbfmaproom2
```

### Launch the development server

```
CONFIG=fbfmaproom-sample.yaml:config-local.yaml python fbfmaproom.py
```

### Test the application in a browser 

On your laptop, connect to e.g. `http://localhost:1234/fbfmaproom/ethiopia`, substituting your chosen port number for `1234`. ssh port forwarding will forward the connection to the application running on the server.

## Updating datasets

All datasets used by the application are stored in zarr format. There are two categories of of datasets: ones that are downloaded (as netcdf) from the Data Library and then converted to zarr, and ones that are converted directly from the original source files to zarr by a python script, without passing through the Data Library.

### Datasets from the Data Library

The script `fbf-update-data.py` pulls data from the DL and saves it as zarr. First read the script and find the name of the dataset you want to update, e.g. `ethiopia/rain-mam`. Then run the script as follows, substituting the chosen dataset name. Note that we run this script in the enactsmaproom conda environment, not the fbfmaproom environment. The fbfmaproom environment is missing the netcdf library, which the script requires; using the enactsmaproom environment is a lazy workaround.
```
conda activate enactsmaproom
CONFIG=fbfmaproom-sample.py python fbf-update-data.py ethiopia/pnep
```
 
### Datasets not from the Data Library

Scripts for converting non-Data Library datasets to zarr are contained in the `data-conversion-scripts` directory. One of these scripts, `zarrify-forecast.py`, is used for all PyCPT forecast datasets, for all countries. Other scripts are specific to a single dataset and are kept in per-country subdirectories.

For `zarrify-forecast.py`, edit the end of the script to indicate which dataset you want to update.

`zarrify-forecast.py` also must be run in the `enactsmaproom` conda environment.

## Adding or removing python dependencies

Edit `environment.yml`, then regenerate the lock files as follows:
```
conda install conda-lock
conda-lock lock -f environment.yml -f environment-dev.yml
conda-lock render
```

## Adding a foreign table

This maproom makes use of foreign tables in Postgres. Here's a brief explanation of how to add one:

1. Login in the pgdb12 server using psql. Make sure the user/role you use has the appropriate permissions

    psql -h pgdb12.iri.columbia.edu -U fist -W DesignEngine

2. Create a server. Note that this has probably already been done so should be unnecessary

    CREATE SERVER dlcomputemon1_iridb FOREIGN DATA WRAPPER postgres_fdw OPTIONS (host 'dlcomputemon1.iri.columbia.edu', port '5432', dbname 'iridb');

3. Import the specific table(s) you want. It is advised to use the command below and not `IMPORT FOREIGN TABLE` so that the
   definition of the table does not have to be (re)specified by hand. The schema can be arbitrarily chosen, I've used public here
   but it might be advisable to create a schema for foreign tables specifically. Make sure you have the right schema of the table in the foreign server.

    IMPORT FOREIGN SCHEMA public LIMIT TO (table_name) FROM SERVER dlcomputemon1_iridb INTO public;

4. Create a user mapping for every user in pgdb12 you want to have access to tables in the foreign server

    CREATE USER MAPPING FOR dero SERVER dlcomputemon1_iridb OPTIONS (user 'ingrid_ro', password 'PASSWORD');

5. Grant select privileges to the local user if necessary

    GRANT SELECT ON table_name TO dero;
