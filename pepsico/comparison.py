import xarray as xr
import os
import re


ds_zarr = '/data/remic/mydatafiles/zarr/test/hurs'
ds_nc_dir = '/Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/historical/MPI-ESM1-2-HR/'

#only using NetCDF files that contain 'hurs' variable
def contains_hurs(filename):
    return 'hurs' in filename

#extracting years from nc file
def extract_years(filename):
    match = re.search(r'(\d{4})_(\d{4})\.nc$', filename)
    if match:
        start_year = int(match.group(1))
        end_year = int(match.group(2))
        return start_year, end_year
    else:
        raise ValueError("Unable to find years")


#function that compares nc files(hurs) vs zarred file
def compare_datasets(nc_file_path, ds_zarr):
    #returns true if datasets are identical, false otherwise

    #open datasets
    zarr_dSet = xr.open_zarr(ds_zarr)
    nc_dSet = xr.open_dataset(nc_file_path)[['hurs']]


    #extracting start and end years from nc file
    start_year, end_year = extract_years(os.path.basename(nc_file_path))
    time_start = f'{start_year}-01-01'
    time_end = f'{end_year}-12-31'

    #check if datasets are empty
    if len(nc_dSet) == 0 or len(zarr_dSet) == 0:
        print("Datasets are empty")
        return False

    
    #mapping from zar to nc dimensions (X,Y,T) vs (lon,lat,time)
    coord_map = {'lon': 'X', 'lat': 'Y', 'time': 'T'}

    #xararay rename function to reset nc labels to zarr
    nc_dSet = nc_dSet.rename({orig_coord: new_coord for orig_coord, new_coord in coord_map.items() if orig_coord in nc_dSet.coords})
    
    nc_dSet_sliced = nc_dSet.sel(T=slice(time_start, time_end))
    zarr_dSet_sliced = zarr_dSet.sel(T=slice(time_start, time_end))
    
    #NYC coordinates. selecting data at specific coordinates using .sel
    nyc_lat, nyc_lon = 40.7128, -74.0060
    nc_data_nyc = nc_dSet_sliced.sel(X=nyc_lon, Y=nyc_lat, method='nearest')['hurs']
    zarr_data_nyc = zarr_dSet_sliced.sel(X=nyc_lon, Y=nyc_lat, method='nearest')['hurs']

    #los gatos, california coordinates. selecting data at specific coordinates
    ca_lat, ca_lon = 37.2266, -121.9737
    nc_data_ca = nc_dSet_sliced.sel(X=ca_lon, Y=ca_lat, method='nearest')['hurs']
    zarr_data_ca = zarr_dSet_sliced.sel(X=ca_lon, Y=ca_lat, method='nearest')['hurs']
    

    #testing.assert_equal is an xarray tool to compare datasets (see if they are identical)
    try:
        xr.testing.assert_equal(nc_dSet_sliced, zarr_dSet_sliced)
        print('Datasets are identical')
        
    except AssertionError as e:
        print("Datasets are not identical.")
        print(e)  # Print the error message associated with the AssertionError
    
    # Find max and min values for nc and zarr datasets at NYC coordinates
    max_nc_nyc = nc_data_nyc.max().values
    max_zarr_nyc = zarr_data_nyc.max().values
    min_nc_nyc = nc_data_nyc.min().values
    min_zarr_nyc = zarr_data_nyc.min().values

    #same, but for CA coordinates
    max_nc_ca = nc_data_ca.max().values
    max_zarr_ca = zarr_data_ca.max().values
    min_nc_ca = nc_data_ca.min().values
    min_zarr_ca = zarr_data_ca.min().values

    # Count occurrences of max and min values in datasets
    max_nc_nyc_count = (nc_data_nyc == max_nc_nyc).sum().values
    min_nc_nyc_count = (nc_data_nyc == min_nc_nyc).sum().values
    max_zarr_nyc_count = (zarr_data_nyc == max_zarr_nyc).sum().values
    min_zarr_nyc_count = (zarr_data_nyc == min_zarr_nyc).sum().values

    max_nc_ca_count = (nc_data_ca == max_nc_ca).sum().values
    min_nc_ca_count = (nc_data_ca == min_nc_ca).sum().values
    max_zarr_ca_count = (zarr_data_ca == max_zarr_ca).sum().values
    min_zarr_ca_count = (zarr_data_ca == min_zarr_ca).sum().values
    
    print(f"Max value in NetCDF file {os.path.basename(nc_file_path)} at NYC coordinates: {max_nc_nyc}")
    print(f"NYC max value (zarr) {max_zarr_nyc}, Number of maxs: {max_nc_nyc_count}")
    print(f"NYC min value (netCDF) {os.path.basename(nc_file_path)} at NYC coordinates: {min_nc_nyc}")
    print(f"NYC min value (zarr): {min_zarr_nyc}, Number of mins: {min_zarr_nyc_count}")

    print(f"Max value in NetCDF file {os.path.basename(nc_file_path)} at CA coordinates: {max_nc_ca}")
    print(f"Max value (zarr): {max_zarr_ca}, Number of maxs: {max_zarr_ca_count}")
    print(f"CA min value {os.path.basename(nc_file_path)} at CA coordinates: {min_nc_ca}")
    print(f"CA min value (zarr): {min_zarr_ca}, Number of mins: {min_zarr_ca_count}")

    # Check if max and min values are identical
    max_identical = (max_nc_nyc == max_zarr_nyc) and (max_nc_ca == max_zarr_ca)
    min_identical = (min_nc_nyc == min_zarr_nyc) and (min_nc_ca == min_zarr_ca)

    if max_identical and min_identical:
        print("Max and min values are identical.")
    else:
        print("Max and min values are not identical.")

    return max_identical and min_identical


#list of all NetCDF files
nc_files = [os.path.join(ds_nc_dir, f) for f in os.listdir(ds_nc_dir) if f.endswith('.nc')]

#filtered list of NetCDF files with hurs
nc_files_with_hurs = [f for f in nc_files if contains_hurs(f)]

print('Comparing datasets and min/max values...')

#comparing datasets
all_identical = True
for nc_file in nc_files_with_hurs:
    if not compare_datasets(nc_file, ds_zarr):
        all_identical = False

#testing data
if all_identical:
    print("NC and Zarr files are identical")
else:
    print("Datasets are not the same ")
