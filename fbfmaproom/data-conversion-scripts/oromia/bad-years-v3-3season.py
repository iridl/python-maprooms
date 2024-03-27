import cftime
import datetime
import pandas as pd
import xarray as xr
import sys
import os
import numpy as np
import shutil

if sys.argv[1] == 'test':
    ds = xr.open_zarr('/data/aaron/fbf-candidate/djibouti/bad-years-v2-jas.zarr')
    print(ds['T'].values)
    print(ds['rank'].values)
    sys.exit()
elif sys.argv[1] == 'test1':
    bad=pd.read_csv('../djibouti/bad_years.csv', skiprows=1)
    print(list(set(bad['ond-nma_bady'])))
    sys.exit()

def convert_zarr_to_dataset(zarr_path):
    # Verifica si el directorio Zarr existe
    if os.path.exists(zarr_path) and os.path.isdir(zarr_path):
        try:
            # Intenta abrir el archivo Zarr como un Dataset de Xarray
            ds = xr.open_zarr(zarr_path)
            #nombres_variables = list(ds.data_vars)
            for nombre_variable, variable in ds.data_vars.items():
                print(nombre_variable)
                print(ds[nombre_variable].values)
            print("Zarr file successfully converted to Xarray Dataset.")
            return ds
        except Exception as e:
            print(f"An error occurred while opening the Zarr file: {e}")
    else:
        print("Zarr directory does not exist.")
        return None

dirout='/data/aaron/fbf-candidate/'
regionsdic = {
    #"ethiopia/southern-oromia": {"file": "bad_years.csv","time-span":'2000-2023'}
    "ethiopia/southern-oromia": {"file": "/data/aaron/fbf-candidate/original-data/oromia/bad_years.csv",
                                 "time-span":'2000-2023',
                                 "version":'v3'
                                 },
  "djibouti": {"file": "/data/aaron/fbf-candidate/original-data/djibouti/bad_years.csv",
               "time-span":None,
               "version":'v2'
               }
    
}

vars=['rank','hum_bady','nma_bady']

seasons = {"djf":1, 
           "jfm":2, 
           "fma":3, 
           "mam":4, 
           "amj":5, 
           "mjj":6, 
           "jja":7, 
           "jas":8, 
           "aso":9, 
           "son":10, 
           "ond":11, 
           "ndj":12}



if  len(sys.argv) < 3:
    print("\033[91m"+"Two arguments is expected. (season, region)"+"\033[0m")
    print("\033[1m"+"Example: python bad-years-v3-3season.py ond djibouti"+"\033[0m")
    sys.exit()
elif not sys.argv[1] in seasons: 
    print("\033[91m"+"The season is not allowed."+"\033[0m")
    print("\033[1m"+"Example: "+", ".join(seasons.keys()) +"\033[0m")
    sys.exit()
elif not sys.argv[2].lower() in regionsdic: 
    print("\033[91m"+"The region is not defined."+"\033[0m")
    print("\033[1m"+"regions available: "+ (", ".join(regionsdic.keys())) +"\033[0m")
    sys.exit()
elif len(sys.argv) ==4:
    if not sys.argv[3] in vars:
        print("\033[91m"+"The var is not allowed."+"\033[0m")
        print("\033[1m"+"Example: "+", ".join(vars) +"\033[0m")
        sys.exit()

    season = f"{sys.argv[1].lower()}"
    region = f"{sys.argv[2]}".lower()
    var = f"{sys.argv[3]}"
    column_file=season+'-'+var
    
else:
    season = f"{sys.argv[1].lower()}"
    region = f"{sys.argv[2]}".lower()
    var = ''
    column_file=season
    

bad=pd.read_csv(regionsdic[region]['file'], skiprows=1)
if not regionsdic[region]['time-span'] == None:
    year_ini, year_end = map(int, regionsdic[region]['time-span'].split('-'))
    for year in range(year_ini, year_end + 1):
        if not year in bad['year'].values:
            new_row = {bad.columns[0]: year, bad.columns[1]: 8}
            new_row_df = pd.DataFrame(new_row, index=[0])
            bad=pd.concat([bad, new_row_df], ignore_index=True)
            #print(f"El año {year} no está presente en la columna 'year'.")
    bad = bad.sort_values(by='year').reset_index(drop=True)
    del year, new_row, new_row_df,year_ini,year_end

#sys.exit()
years = [cftime.Datetime360Day(y, seasons[season], 16) for y in bad['year'].tolist()]

if column_file in bad.columns:
    ranks = bad[column_file].to_list()
    if var=='nma_bady':
        ranks=[3 if rank == 'Moderate Impact' else 
               2 if rank == 'Normal' else 
               1 if rank == 'High Impact' else 
               rank for rank in ranks]
else:
    print("\033[91m"+"The season "+column_file+" is not available in the file "+regionsdic[region]['file']+"."+"\033[0m")
    sys.exit()

if not var =='':
    ds = xr.Dataset(data_vars={var: xr.DataArray(ranks, coords={"T": years})})
else:  
    ds = xr.Dataset(data_vars={"rank": xr.DataArray(ranks, coords={"T": years})})

if "/" in region:
    file=os.path.join(dirout,region+'-bad-years-'+regionsdic[region]['version']+'-'+season+'.zarr')
else:
    file=os.path.join(dirout,region,'bad-years-'+regionsdic[region]['version']+'-'+season+'.zarr')

ds_aux = convert_zarr_to_dataset(file)
if ds_aux is not None:
    # El Dataset está listo para ser utilizado
    if np.array_equal(ds_aux['T'].values, ds['T'].values): 
        #print('Coordenada es la misma')
        shutil.move(file, file.replace('.zarr','OLD.zarr'))
        ds_aux[var] = (('T'), ranks)
        ds_aux.to_zarr(file)

    else:
        print("\033[91m"+"The T coordinate has different sizes."+"\033[0m")
        sys.exit()
    print(ds_aux)
else:
    # El archivo Zarr no existe o hubo un error
    #print("Operation failed.")
    ds.to_zarr(file)
sys.exit()

ds.to_zarr(file)
