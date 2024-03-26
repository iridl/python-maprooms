import cftime
import datetime
import pandas as pd
import xarray as xr
import sys
import os

dirout='/data/aaron/fbf-candidate/'
regionsdic = {
        "oromia": {"file": "/data/aaron/fbf-candidate/original-data/oromia/bad_years.csv"}
}

seasons=["DJF","JFM","FMA","MAM","AMJ","MJJ","JJA","JAS","ASO","SON","OND","NDJ"]

if  len(sys.argv) < 3:
    print("\033[91m"+"Two arguments is expected. (season, region)"+"\033[0m")
    print("\033[1m"+"Example: python bad-years-v3-3season.py ond oromia"+"\033[0m")
    sys.exit()
elif not sys.argv[1].upper() in seasons: 
    print("\033[91m"+"The station is not allowed."+"\033[0m")
    print("\033[1m"+"Example: "+", ".join(list(map(str.lower,seasons))) +"\033[0m")
    sys.exit()
elif not sys.argv[2].lower() in regionsdic: 
    print("\033[91m"+"The region is not defined."+"\033[0m")
    print("\033[1m"+"regions avalibale: "+ (", ".join(regionsdic.keys())) +"\033[0m")
    sys.exit()
else:
    season = sys.argv[1].lower()
    region = f"{sys.argv[2]}".lower()
    

bad=pd.read_csv(regionsdic[region]['file'], skiprows=1)
years = [cftime.Datetime360Day(y + 1, 1, 16) for y in bad['year'].tolist()]
if season in bad.columns:
    ranks = bad[season].to_list()
else:
    print("\033[91m"+"The station "+season+" is not available in the file "+regionsdic[region]['file']+"."+"\033[0m")
    sys.exit()


ds = xr.Dataset(data_vars={"rank": xr.DataArray(ranks, coords={"T": years})})

ds.to_zarr(os.path.join(dirout,region,'bad-years-v3-'+season+'.zarr'))
