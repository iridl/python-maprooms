import cftime
import datetime
import pandas as pd
import xarray as xr
import sys
import os


dirout='/data/aaron/fbf-candidate/'
regionsdic = {
    "ethiopia/southern-oromia": {"file": "/data/aaron/fbf-candidate/original-data/oromia/bad_years.csv"},
  "djibouti": {"file": "/data/aaron/fbf-candidate/original-data/djibouti/bad_years.csv"}
    
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
years = [cftime.Datetime360Day(y, seasons[season], 16) for y in bad['year'].tolist()]
if season in bad.columns:
    ranks = bad[season].to_list()
else:
    print("\033[91m"+"The season "+season+" is not available in the file "+regionsdic[region]['file']+"."+"\033[0m")
    sys.exit()

ds = xr.Dataset(data_vars={"rank": xr.DataArray(ranks, coords={"T": years})})

ds.to_zarr(os.path.join(dirout,region,'bad-years-v3-'+season+'.zarr'))
