# Setting up a development FBF Maproom

These instructions are specific to developers working on an IRI server.

## Log into the server, with port forwarding enabled

### Valid servers are

- shortfin.iri.columbia.edu
- storm.iri.columbia.edu

Choose port on which to run your development server. You must choose one that isn't in use by another developer on the same server. Log into the server and forward your chosen port (1234 in the example, but don’t use this number make up your own random number).

```
ssh -L 1234:localhost:1234 shortfin.iri.columbia.edu
```

## GitHub Account

The repository is public, so unless you are making changes to the code, you don't need a github account.

### On the server, create an ssh key to use with GitHub if you don't already have one.

```
ssh-keygen -f ~/.ssh/id_rsa_github -t rsa
```

Provide a strong password when prompted. Then

```
cat ~/.ssh/id_rsa_github.pub
```

and copy the output of that command for use in the next one. It should start with `ssh-rsa`.

### Configure GitHub to use that ssh key

In a browser, visit [https://github.com/settings/ssh/new](https://github.com/settings/ssh/new). For the title put e.g. "IRI development server," and in the Key box paste the output from the previous comand. Then click the green "Add SSH key" button.

### Configure your IRI account to use that ssh key (Mac, Linux)

```
cat << EOF >> ~/.ssh/config

Host github.com
HostName github.com
IdentityFile ~/.ssh/id_rsa_github
PreferredAuthentications publickey
IdentitiesOnly yes

EOF
chmod 600 ~/.ssh/config
```

## Clone this repository and enter the fbfmaproom directory

If you have a github account and have set up your ssh key:

```
git clone git@github.com:iridl/python-maprooms
cd python-maprooms/fbfmaproom
```

If you don't have a github account and will not be making any code changes:

```
git clone https://github.com/iridl/python-maprooms.git
cd python-maprooms/fbfmaproom
```

## Python miniconda environment

### Enable conda

This could be added to your ~/.bash_profile to be loaded every time you login.

```
module load python/miniconda3
source /software/rhel8/x86_64/miniconda/3/etc/profile.d/conda.sh
```

### Optionally create a conda environment containing this project's dependencies

On the IRI servers, this environment is pre-built, so skip this step unless you're using an environment where this doesn't exist.

```
cond env list
```

If fbmaproom2 doesn't exist:

```
conda create -n fbfmaproom2 --file conda-linux-64.lock
```

(substituting conda-osx-64.lock or conda-win-64.lock appropriate)

Do not use conda-lock for this.

Note that the command is `conda create`, not `conda env create`. Both exist, and they're different :-(

### Create a local configuration file

Create a file called `config-local.yaml` with the following contents, modified as noted in the comments.

```
db:
    password: itsasecret # get the real db password from another developer
dev_server_port: 1234 # the port you chose earlier
```

## Running the application in your development environment

### Activate the conda environment

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
conda activate /home/aaron/miniconda3/envs/enactsmaproom
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

1.  Login in the pgdb12 server using psql. Make sure the user/role you use has the appropriate permissionspsql -h pgdb12.iri.columbia.edu -U fist -W DesignEngine
2.  Create a server. Note that this has probably already been done so should be unnecessaryCREATE SERVER dlcomputemon1\_iridb FOREIGN DATA WRAPPER postgres\_fdw OPTIONS (host 'dlcomputemon1.iri.columbia.edu', port '5432', dbname 'iridb');
3.  Import the specific table(s) you want. It is advised to use the command below and not `IMPORT FOREIGN TABLE` so that the  
    definition of the table does not have to be (re)specified by hand. The schema can be arbitrarily chosen, I've used public here  
    but it might be advisable to create a schema for foreign tables specifically. Make sure you have the right schema of the table in the foreign server.IMPORT FOREIGN SCHEMA public LIMIT TO (table\_name) FROM SERVER dlcomputemon1\_iridb INTO public;
4.  Create a user mapping for every user in pgdb12 you want to have access to tables in the foreign serverCREATE USER MAPPING FOR dero SERVER dlcomputemon1\_iridb OPTIONS (user 'ingrid\_ro', password 'PASSWORD');
5.  Grant select privileges to the local user if necessaryGRANT SELECT ON table\_name TO dero;
