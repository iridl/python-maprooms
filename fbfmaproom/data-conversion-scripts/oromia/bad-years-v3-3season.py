import cftime
import datetime
import pandas as pd
import xarray as xr
import sys
import os

#ds = xr.open_zarr('/data/aaron/fbf-candidate/ethiopia/southern-oromia/bad-years-v3-ond.zarr')
#print(ds['T'].values)
#print(ds['rank'].values)
#sys.exit()

dirout='/data/aaron/fbf-candidate/'
regionsdic = {
    #"ethiopia/southern-oromia": {"file": "bad_years.csv","time-span":'2000-2023'}
    "ethiopia/southern-oromia": {"file": "/data/aaron/fbf-candidate/original-data/oromia/bad_years.csv",
                                 "time-span":'2000-2023'
                                 },
  "djibouti": {"file": "/data/aaron/fbf-candidate/original-data/djibouti/bad_years.csv",
               "time-span":None
               }
    
}


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
else:
    season = sys.argv[1].lower()
    region = f"{sys.argv[2]}".lower()
    

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
if season in bad.columns:
    ranks = bad[season].to_list()
else:
    print("\033[91m"+"The season "+season+" is not available in the file "+regionsdic[region]['file']+"."+"\033[0m")
    sys.exit()

ds = xr.Dataset(data_vars={"rank": xr.DataArray(ranks, coords={"T": years})})

if "/" in region:
    file=os.path.join(dirout,region+'-bad-years-v3-'+season+'.zarr')
else:
    file=os.path.join(dirout,region,'bad-years-v3-'+season+'.zarr')

ds.to_zarr(file)
