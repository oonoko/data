#!/usr/bin/env python3
"""
Create realistic synthetic weather data with geographic and climate variations
"""

import pandas as pd
import numpy as np
from scipy.spatial import distance

# Read existing weather data
existing = pd.read_csv('weather_omnogovi_monthly_clean.csv')
print(f"Existing data: {len(existing['soum'].unique())} soums")

# Read full soum list
soums_full = pd.read_csv('omnogovi_soums_full.csv')

# Find missing soums
existing_soums = existing['soum'].unique()
missing_soums = soums_full[~soums_full['soum'].isin(existing_soums)]

print(f"\nMissing soums: {len(missing_soums)}")
print(missing_soums[['soum', 'lat', 'lon']])

# Geographic factors for climate variation
def get_climate_factors(lat, lon):
    """
    Calculate climate adjustment factors based on geography
    - Northern soums: colder, more snow
    - Southern soums: warmer, less precipitation
    - Eastern soums: more continental (extreme temps)
    - Western soums: more moderate
    - Elevation proxy: latitude (higher = colder)
    """
    # Normalize coordinates (Omnogovi range)
    lat_norm = (lat - 42.0) / (46.0 - 42.0)  # 0 = south, 1 = north
    lon_norm = (lon - 100.0) / (107.0 - 100.0)  # 0 = west, 1 = east
    
    factors = {
        # Temperature adjustments
        'temp_offset': -3.0 * lat_norm,  # Colder in north
        'temp_range': 1.5 * lon_norm,    # More extreme in east
        
        # Precipitation adjustments
        'precip_factor': 0.7 + 0.3 * lat_norm,  # More in north
        'snow_factor': 1.0 + 0.5 * lat_norm,    # More snow in north
        
        # Wind adjustments
        'wind_factor': 1.0 + 0.2 * (1 - lat_norm),  # Windier in south
        
        # Seasonal variation
        'winter_severity': 1.0 + 0.3 * lat_norm,  # Harsher winter in north
        'summer_heat': 1.0 + 0.2 * (1 - lat_norm)  # Hotter summer in south
    }
    
    return factors

# Create realistic synthetic data
synthetic_data = []

for _, missing_soum in missing_soums.iterrows():
    print(f"\n{'='*60}")
    print(f"Creating data for {missing_soum['soum']}")
    print(f"Location: {missing_soum['lat']:.2f}°N, {missing_soum['lon']:.2f}°E")
    
    # Get climate factors
    factors = get_climate_factors(missing_soum['lat'], missing_soum['lon'])
    print(f"Climate factors:")
    print(f"  Temp offset: {factors['temp_offset']:.2f}°C")
    print(f"  Precip factor: {factors['precip_factor']:.2f}x")
    print(f"  Snow factor: {factors['snow_factor']:.2f}x")
    
    # Find 3 nearest soums
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
    
    distances = sorted(distances, key=lambda x: x['distance'])[:3]
    print(f"\nNearest soums:")
    for d in distances:
        print(f"  - {d['soum']}: {d['distance']:.2f}°")
    
    # Inverse distance weights
    total_weight = sum(1/d['distance'] for d in distances)
    weights = [1/d['distance']/total_weight for d in distances]
    
    # Get data from nearest soums
    nearest_data = []
    for d in distances:
        soum_data = existing[existing['soum'] == d['soum']].copy()
        nearest_data.append(soum_data)
    
    # Create synthetic data with realistic variations
    for year in range(2015, 2025):
        # Annual variation (some years colder/warmer)
        annual_temp_var = np.random.normal(0, 1.5)
        annual_precip_var = np.random.uniform(0.8, 1.2)
        
        for month in range(1, 13):
            # Seasonal factors
            is_winter = month in [11, 12, 1, 2, 3]
            is_summer = month in [6, 7, 8]
            
            # Get weighted average from nearest soums
            values = {
                'avg_temp': 0,
                'min_temp': 0,
                'wind_speed': 0,
                'snowfall_sum': 0,
                'precip_sum': 0
            }
            
            for i, soum_data in enumerate(nearest_data):
                row = soum_data[(soum_data['year'] == year) & (soum_data['month'] == month)]
                if len(row) > 0:
                    values['avg_temp'] += row['avg_temp'].values[0] * weights[i]
                    values['min_temp'] += row['min_temp'].values[0] * weights[i]
                    values['wind_speed'] += row['wind_speed'].values[0] * weights[i]
                    values['snowfall_sum'] += row['snowfall_sum'].values[0] * weights[i]
                    values['precip_sum'] += row['precip_sum'].values[0] * weights[i]
            
            # Apply geographic adjustments
            if is_winter:
                # Winter adjustments
                values['avg_temp'] += factors['temp_offset'] * factors['winter_severity']
                values['min_temp'] += factors['temp_offset'] * factors['winter_severity'] * 1.2
                values['snowfall_sum'] *= factors['snow_factor']
                values['wind_speed'] *= factors['wind_factor']
            elif is_summer:
                # Summer adjustments
                values['avg_temp'] += factors['temp_offset'] * 0.5 + annual_temp_var
                values['avg_temp'] *= factors['summer_heat']
                values['min_temp'] += factors['temp_offset'] * 0.3
            else:
                # Spring/Fall
                values['avg_temp'] += factors['temp_offset'] * 0.7 + annual_temp_var * 0.5
                values['min_temp'] += factors['temp_offset'] * 0.8
            
            # Precipitation adjustments
            values['precip_sum'] *= factors['precip_factor'] * annual_precip_var
            
            # Add random variation (weather is chaotic!)
            values['avg_temp'] += np.random.normal(0, 0.8)
            values['min_temp'] += np.random.normal(0, 1.2)
            values['wind_speed'] *= np.random.uniform(0.9, 1.1)
            values['snowfall_sum'] *= np.random.uniform(0.8, 1.3) if values['snowfall_sum'] > 0 else 0
            values['precip_sum'] *= np.random.uniform(0.7, 1.4)
            
            # Ensure realistic bounds
            values['wind_speed'] = max(0, values['wind_speed'])
            values['snowfall_sum'] = max(0, values['snowfall_sum'])
            values['precip_sum'] = max(0, values['precip_sum'])
            
            # Winter snow logic
            if is_winter and values['avg_temp'] < -5:
                # Some precipitation becomes snow
                if values['precip_sum'] > 0 and values['snowfall_sum'] == 0:
                    values['snowfall_sum'] = values['precip_sum'] * 0.3
            
            synthetic_data.append({
                'aimag': missing_soum['aimag'],
                'soum': missing_soum['soum'],
                'lat': missing_soum['lat'],
                'lon': missing_soum['lon'],
                'year': year,
                'month': month,
                'avg_temp': round(values['avg_temp'], 3),
                'min_temp': round(values['min_temp'], 3),
                'wind_speed': round(values['wind_speed'], 3),
                'snowfall_sum': round(values['snowfall_sum'], 2),
                'precip_sum': round(values['precip_sum'], 2)
            })
    
    print(f"✓ Created {len([d for d in synthetic_data if d['soum'] == missing_soum['soum']])} months")

# Combine with existing data
if synthetic_data:
    synthetic_df = pd.DataFrame(synthetic_data)
    combined = pd.concat([existing, synthetic_df], ignore_index=True)
    
    # Save
    output_file = 'weather_omnogovi_monthly_clean.csv'
    combined.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Saved to {output_file}")
    print(f"Total soums: {len(combined['soum'].unique())}")
    print(f"Total rows: {len(combined)}")
    
    # Show statistics comparison
    print(f"\n{'='*60}")
    print("Climate comparison (January average):")
    print(f"{'Soum':<20} {'Avg Temp':<10} {'Min Temp':<10} {'Snow':<10} {'Source'}")
    print("-" * 60)
    
    jan_data = combined[combined['month'] == 1].groupby('soum').agg({
        'avg_temp': 'mean',
        'min_temp': 'mean',
        'snowfall_sum': 'mean'
    }).round(1)
    
    for soum in sorted(combined['soum'].unique()):
        if soum in jan_data.index:
            row = jan_data.loc[soum]
            source = "real" if soum in existing_soums else "synthetic"
            print(f"{soum:<20} {row['avg_temp']:>8.1f}°C {row['min_temp']:>9.1f}°C {row['snowfall_sum']:>8.1f}mm {source}")
    
else:
    print("\n⚠️  No synthetic data created")
