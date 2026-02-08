#!/usr/bin/env python3
"""
Create synthetic weather data for missing soums based on nearby soums
"""

import pandas as pd
import numpy as np
from scipy.spatial import distance

# Read existing weather data
existing = pd.read_csv('weather_omnogovi_monthly_clean.csv')
print(f"Existing data: {len(existing['soum'].unique())} soums")
print(existing['soum'].unique())

# Read full soum list with coordinates
soums_full = pd.read_csv('omnogovi_soums_full.csv')

# Find missing soums
existing_soums = existing['soum'].unique()
missing_soums = soums_full[~soums_full['soum'].isin(existing_soums)]

print(f"\nMissing soums: {len(missing_soums)}")
print(missing_soums[['soum', 'lat', 'lon']])

# Create synthetic data for missing soums
synthetic_data = []

for _, missing_soum in missing_soums.iterrows():
    print(f"\nCreating synthetic data for {missing_soum['soum']}...")
    
    # Find 3 nearest soums with data
    distances = []
    for _, existing_soum in soums_full[soums_full['soum'].isin(existing_soums)].iterrows():
        dist = distance.euclidean(
            [missing_soum['lat'], missing_soum['lon']],
            [existing_soum['lat'], existing_soum['lon']]
        )
        distances.append({
            'soum': existing_soum['soum'],
            'distance': dist,
            'lat': existing_soum['lat'],
            'lon': existing_soum['lon']
        })
    
    # Sort by distance and take top 3
    distances = sorted(distances, key=lambda x: x['distance'])[:3]
    
    print(f"  Nearest soums:")
    for d in distances:
        print(f"    - {d['soum']}: {d['distance']:.2f}°")
    
    # Calculate weights (inverse distance)
    total_weight = sum(1/d['distance'] for d in distances)
    weights = [1/d['distance']/total_weight for d in distances]
    
    print(f"  Weights: {[f'{w:.2f}' for w in weights]}")
    
    # Get data from nearest soums
    nearest_data = []
    for d in distances:
        soum_data = existing[existing['soum'] == d['soum']].copy()
        nearest_data.append(soum_data)
    
    # Create synthetic data by weighted average
    for year in range(2015, 2025):
        for month in range(1, 13):
            # Get values from each nearest soum
            values = {
                'avg_temp': [],
                'min_temp': [],
                'wind_speed': [],
                'snowfall_sum': [],
                'precip_sum': []
            }
            
            for i, soum_data in enumerate(nearest_data):
                row = soum_data[(soum_data['year'] == year) & (soum_data['month'] == month)]
                if len(row) > 0:
                    for key in values.keys():
                        values[key].append(row[key].values[0] * weights[i])
            
            # Calculate weighted average
            if values['avg_temp']:
                synthetic_data.append({
                    'aimag': missing_soum['aimag'],
                    'soum': missing_soum['soum'],
                    'lat': missing_soum['lat'],
                    'lon': missing_soum['lon'],
                    'year': year,
                    'month': month,
                    'avg_temp': sum(values['avg_temp']),
                    'min_temp': sum(values['min_temp']),
                    'wind_speed': sum(values['wind_speed']),
                    'snowfall_sum': sum(values['snowfall_sum']),
                    'precip_sum': sum(values['precip_sum'])
                })
    
    print(f"  ✓ Created {len([d for d in synthetic_data if d['soum'] == missing_soum['soum']])} months")

# Combine with existing data
if synthetic_data:
    synthetic_df = pd.DataFrame(synthetic_data)
    combined = pd.concat([existing, synthetic_df], ignore_index=True)
    
    # Save
    output_file = 'weather_omnogovi_monthly_complete.csv'
    combined.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved to {output_file}")
    print(f"Total soums: {len(combined['soum'].unique())}")
    print(f"Total rows: {len(combined)}")
    
    print(f"\nAll soums:")
    for soum in sorted(combined['soum'].unique()):
        count = len(combined[combined['soum'] == soum])
        source = "real" if soum in existing_soums else "synthetic"
        print(f"  - {soum}: {count} months ({source})")
    
    # Update the main file
    combined.to_csv('weather_omnogovi_monthly_clean.csv', index=False)
    print(f"\n✅ Updated weather_omnogovi_monthly_clean.csv")
    
else:
    print("\n⚠️  No synthetic data created")
