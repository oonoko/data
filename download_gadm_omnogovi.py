#!/usr/bin/env python3
"""
Download and extract Omnogovi soums from GADM data
"""

import requests
import json
import geopandas as gpd

print("Downloading GADM Mongolia ADM2 (soum level) data...")

# GADM Mongolia ADM2 GeoJSON
url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_MNG_2.json.zip"

try:
    # Download
    print(f"Downloading from {url}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    
    # Save zip
    with open('gadm41_MNG_2.json.zip', 'wb') as f:
        f.write(response.content)
    
    print(f"✓ Downloaded {len(response.content)} bytes")
    
    # Extract and read with geopandas
    print("\nReading GeoJSON...")
    gdf = gpd.read_file('zip://gadm41_MNG_2.json.zip')
    
    print(f"Total features: {len(gdf)}")
    print(f"Columns: {gdf.columns.tolist()}")
    
    # Filter Omnogovi
    omnogovi = gdf[gdf['NAME_1'] == 'Ömnögovi'].copy()
    
    print(f"\nÖmnögovi soums: {len(omnogovi)}")
    print("\nSoum names:")
    for idx, row in omnogovi.iterrows():
        print(f"  - {row['NAME_2']}")
    
    # Save as GeoJSON
    output_file = 'omnogovi_soums.geojson'
    omnogovi.to_file(output_file, driver='GeoJSON')
    
    print(f"\n✅ Saved to {output_file}")
    
    # Also save simplified version (smaller file)
    omnogovi_simple = omnogovi.copy()
    omnogovi_simple['geometry'] = omnogovi_simple['geometry'].simplify(0.01)
    
    output_simple = 'omnogovi_soums_simple.geojson'
    omnogovi_simple.to_file(output_simple, driver='GeoJSON')
    
    print(f"✅ Saved simplified version to {output_simple}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("\nAlternative: Using existing MNG_ADM2.geojson...")
    
    try:
        # Try to read existing file
        with open('MNG_ADM2.geojson', 'r') as f:
            data = json.load(f)
        
        print(f"Total features: {len(data['features'])}")
        
        # Filter Omnogovi (try different name variations)
        omnogovi_features = []
        omnogovi_names = ['omnogovi', 'ömnögovi', 'umnugovi', 'south gobi']
        
        for feature in data['features']:
            props = feature.get('properties', {})
            # Check various property names
            for key in props:
                value = str(props[key]).lower()
                if any(name in value for name in omnogovi_names):
                    omnogovi_features.append(feature)
                    print(f"  Found: {props.get('shapeName', 'Unknown')}")
                    break
        
        if omnogovi_features:
            omnogovi_geojson = {
                'type': 'FeatureCollection',
                'features': omnogovi_features
            }
            
            with open('omnogovi_soums.geojson', 'w') as f:
                json.dump(omnogovi_geojson, f)
            
            print(f"\n✅ Extracted {len(omnogovi_features)} Omnogovi features")
        else:
            print("\n⚠️  No Omnogovi features found")
            
    except Exception as e2:
        print(f"✗ Also failed: {e2}")
