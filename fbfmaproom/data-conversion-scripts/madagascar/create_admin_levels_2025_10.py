#!/usr/bin/env python3
"""
Create Admin 0, Admin 1, and Admin 2 Shapefiles from Admin 2 Input

This script processes Madagascar administrative boundary data to create
four separate shapefiles for different administrative levels:
- Admin 0: National boundaries (country level)
- Admin 1: Regional boundaries (province/region level)
- Admin 2: District boundaries (district level)
- ALL: Combined shapefile with all admin levels

The script:
1. Dissolves (merges) lower-level boundaries to create higher-level boundaries
2. Reprojects from projected coordinate system (EPSG:3395 - meters) to 
   geographic coordinate system (EPSG:4326 - degrees) for web mapping compatibility
3. Maintains hierarchical field structure (e.g., Admin 1 includes Admin 0 fields)
4. Creates an ALL levels shapefile with admLevel field for filtering

Requirements:
    - geopandas
    - pandas
    - zipfile (standard library)
    - tempfile (standard library)
    - pathlib (standard library)

Usage:
    python create_admin_levels.py

Input:
    SHP_ADM2_UPDATE_dante.zip - Original admin 2 shapefile in projected CRS (EPSG:3395)

Output:
    SHP_ADM2_UPDATE.zip - New zip file containing all four admin level shapefiles
    in geographic coordinate system (EPSG:4326) for Python maproom integration
"""

import geopandas as gpd
import pandas as pd
import zipfile
import tempfile
import os
import shutil
from pathlib import Path

def main():
    """
    Main function to process shapefiles and create admin levels.
    
    Process:
    1. Extract the original admin 2 shapefile from the input zip
    2. Read the shapefile using GeoPandas
    3. Reproject from EPSG:3395 (projected, meters) to EPSG:4326 (geographic, degrees)
    4. Create admin 0 by dissolving all polygons into one national boundary
    5. Create admin 1 by dissolving polygons grouped by region code (includes Admin 0 fields)
    6. Create admin 2 by filtering columns from the original data (includes Admin 0 + Admin 1 fields)
    7. Create ALL levels shapefile by combining all admin levels with admLevel field
    8. Save all four shapefiles with consistent naming
    9. Package everything into a new zip file with geographic coordinates
    """
    # Input and output paths
    input_zip = "SHP_ADM2_UPDATE_dante.zip"
    output_zip = "SHP_ADM2_UPDATE.zip"
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print("Extracting shapefile from zip...")
        # Extract the shapefile from zip
        with zipfile.ZipFile(input_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_path)
        
        # Read the admin 2 shapefile
        shapefile_path = temp_path / "SHP_ADM2_UPDATE" / "mdg_bnd_adm2_dis_pam.shp"
        print(f"Reading shapefile: {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)
        
        print(f"Original shapefile has {len(gdf)} features")
        print(f"Columns: {list(gdf.columns)}")
        print(f"Original CRS: {gdf.crs}")
        
        # Reproject from projected coordinate system to geographic coordinate system
        # Input: EPSG:3395 (WGS 84 / World Mercator - projected, units in meters)
        # Output: EPSG:4326 (WGS 84 - geographic, units in degrees)
        # This conversion is required for Python maproom applications which expect
        # geographic coordinates in degrees (lat/lon) rather than projected meters
        print(f"\nReprojecting from {gdf.crs} to EPSG:4326 (WGS 84 - geographic)...")
        gdf = gdf.to_crs('EPSG:4326')
        print(f"Reprojected CRS: {gdf.crs}")
        print("Coordinates are now in degrees (longitude, latitude)")
        
        # Create admin 0 (National) - dissolve by ADM0_PCODE
        # This merges all 120 district polygons into 1 national boundary
        print("\nCreating Admin 0 (National) shapefile...")
        adm0 = gdf.dissolve(by='ADM0_PCODE', aggfunc='first')
        # Reset index to make ADM0_PCODE a column again, then select fields
        adm0 = adm0.reset_index()
        adm0 = adm0[['ADM0_PCODE', 'ADM0_EN', 'geometry']]
        print(f"Admin 0 has {len(adm0)} feature(s)")
        
        # Create admin 1 (Regional) - dissolve by ADM1_PCODE
        # This merges district polygons that share the same region code
        # Include hierarchical fields (Admin 0 + Admin 1 fields)
        print("\nCreating Admin 1 (Regional) shapefile...")
        adm1 = gdf.dissolve(by='ADM1_PCODE', aggfunc='first')
        # Reset index to make ADM1_PCODE a column again, then select fields
        adm1 = adm1.reset_index()
        adm1 = adm1[['ADM0_PCODE', 'ADM0_EN', 'ADM1_PCODE', 'ADM1_EN', 'ADM1_TYPE', 'geometry']]
        print(f"Admin 1 has {len(adm1)} feature(s)")
        
        # Create admin 2 (District) - keep original geometries but filter columns
        # This preserves all 120 district boundaries
        # Include hierarchical fields (Admin 0 + Admin 1 + Admin 2 fields)
        print("\nCreating Admin 2 (District) shapefile...")
        adm2 = gdf[['ADM0_PCODE', 'ADM0_EN', 'ADM1_PCODE', 'ADM1_EN', 'ADM1_TYPE', 'ADM2_PCODE', 'ADM2_EN', 'ADM2_TYPE', 'geometry']].copy()
        print(f"Admin 2 has {len(adm2)} feature(s)")
        
        # Create ALL levels shapefile - combine all admin levels with admLevel field
        # This creates a single shapefile containing all administrative levels,
        # distinguished by the admLevel field (0=National, 1=Regional, 2=District)
        # This format matches the structure used in reference datasets (e.g., Djibouti)
        print("\nCreating ALL levels shapefile...")
        # Add admLevel field to each dataframe to distinguish admin levels
        adm0_all = adm0.copy()
        adm0_all['admLevel'] = 0  # National level
        
        adm1_all = adm1.copy()
        adm1_all['admLevel'] = 1  # Regional level
        
        adm2_all = adm2.copy()
        adm2_all['admLevel'] = 2  # District level
        
        # Combine all levels into a single GeoDataFrame
        all_levels = gpd.GeoDataFrame(pd.concat([adm0_all, adm1_all, adm2_all], ignore_index=True))
        print(f"ALL levels has {len(all_levels)} feature(s) (1 National + {len(adm1_all)} Regional + {len(adm2_all)} District)")
        
        # Save shapefiles to temporary directory
        print("\nSaving shapefiles...")
        adm0_path = temp_path / "mdg_adm0.shp"
        adm1_path = temp_path / "mdg_adm1.shp"
        adm2_path = temp_path / "mdg_adm2.shp"
        all_path = temp_path / "mdg_adm_ALL.shp"
        
        adm0.to_file(adm0_path)
        adm1.to_file(adm1_path)
        adm2.to_file(adm2_path)
        all_levels.to_file(all_path)
        
        print(f"Saved Admin 0: {adm0_path}")
        print(f"Saved Admin 1: {adm1_path}")
        print(f"Saved Admin 2: {adm2_path}")
        print(f"Saved ALL levels: {all_path}")
        
        # Create new zip file with all four shapefiles
        print(f"\nCreating new zip file: {output_zip}")
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
            # Add all shapefile components for each admin level
            for admin_level in ['mdg_adm0', 'mdg_adm1', 'mdg_adm2', 'mdg_adm_ALL']:
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx', '.shp.xml']:
                    file_path = temp_path / f"{admin_level}{ext}"
                    if file_path.exists():
                        zip_ref.write(file_path, file_path.name)
                        print(f"  Added: {file_path.name}")
        
        print(f"\nSuccessfully created {output_zip} with all admin levels!")
        print("\nLayer names:")
        print("- Admin 0 (National): mdg_adm0")
        print("- Admin 1 (Regional): mdg_adm1") 
        print("- Admin 2 (District): mdg_adm2")
        print("- ALL levels: mdg_adm_ALL")
        print(f"\nAll shapefiles are in EPSG:4326 (WGS 84 - geographic coordinate system)")
        print("Coordinates are in degrees (longitude, latitude) for web mapping compatibility")

if __name__ == "__main__":
    main()
