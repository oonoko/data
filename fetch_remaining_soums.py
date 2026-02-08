#!/usr/bin/env python3
"""
Fetch weather data for remaining Omnogovi soums
"""

import pandas as pd
import requests
from datetime import datetime
import time

# Read full soum list
soums_full = pd.read_csv('omnogovi_soums_full.csv')

# Read existing weather data to see which soums we already have
existing_weather = pd.read_csv('weather_omnogovi_monthly_clean.csv')
existing_soums = existing_weather['soum'].unique()

print(f"Existing soums: {len(existing_soums)}")
print(existing_soums)

# Find missing soums
missing_soums = soums_full[~soums_full['soum'].isin(existing_soums)]

print(f"\nMissing soums: {len(missing_soums)}")
print(missing_soums[['soum', 'lat', 'lon']])

# Fetch weather for missing soums
all_data = []

for _, soum in missing_soums.iterrows():
    print(f"\nFetching weather for {soum['soum']}...")
    
    lat = soum['lat']
    lon = soum['lon']
    
    # Open-Meteo API
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2015-01-01",
        "end_date": "2024-12-31",
        "daily": "temperature_2m_mean,temperature_2m_min,wind_speed_10m_max,snowfall_sum,precipitation_sum",
        "timezone": "Asia/Ulaanbaatar"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Parse daily data
        daily = data['daily']
        dates = pd.to_datetime(daily['time'])
        
        df = pd.DataFrame({
            'aimag': soum['aimag'],
            'soum': soum['soum'],
            'lat': lat,
            'lon': lon,
            'date': dates,
            'year': dates.year,
            'month': dates.month,
            'avg_temp': daily['temperature_2m_mean'],
            'min_temp': daily['temperature_2m_min'],
            'wind_speed': daily['wind_speed_10m_max'],
            'snowfall_sum': daily['snowfall_sum'],
            'precip_sum': daily['precipitation_sum']
        })
        
        # Convert to monthly
        monthly = df.groupby(['aimag', 'soum', 'lat', 'lon', 'year', 'month']).agg({
            'avg_temp': 'mean',
            'min_temp': 'min',
            'wind_speed': 'mean',
            'snowfall_sum': 'sum',
            'precip_sum': 'sum'
        }).reset_index()
        
        all_data.append(monthly)
        
        print(f"  ✓ Fetched {len(monthly)} months")
        
        # Be nice to the API
        time.sleep(1)
        
    except Exception as e:
        print(f"  ✗ Error: {e}")

# Combine with existing data
if all_data:
    new_data = pd.concat(all_data, ignore_index=True)
    combined = pd.concat([existing_weather, new_data], ignore_index=True)
    
    # Save
    output_file = 'weather_omnogovi_monthly_all.csv'
    combined.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved to {output_file}")
    print(f"Total soums: {len(combined['soum'].unique())}")
    print(f"Total rows: {len(combined)}")
    print(f"\nSoums:")
    for soum in sorted(combined['soum'].unique()):
        count = len(combined[combined['soum'] == soum])
        print(f"  - {soum}: {count} months")
else:
    print("\n⚠️  No new data fetched")
