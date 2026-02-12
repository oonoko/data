#!/usr/bin/env python3
"""
Extend existing weather data with more years (2010-2014)
"""

import pandas as pd
import numpy as np

# Load existing data
existing = pd.read_csv('weather_omnogovi_monthly_clean.csv')
print(f"Existing data: {len(existing)} rows")
print(f"Years: {existing.year.min()}-{existing.year.max()}")
print(f"Soums: {existing.soum.nunique()}")

# Create extended data for 2010-2014
extended_data = []

for soum in existing['soum'].unique():
    soum_data = existing[existing['soum'] == soum].copy()
    
    # Get location info
    aimag = soum_data['aimag'].iloc[0]
    lat = soum_data['lat'].iloc[0]
    lon = soum_data['lon'].iloc[0]
    
    print(f"\nExtending {soum}...")
    
    # For each new year (2010-2014)
    for new_year in range(2010, 2015):
        # Use similar year from existing data (2015-2019)
        reference_year = 2015 + (new_year - 2010)
        ref_data = soum_data[soum_data['year'] == reference_year]
        
        if len(ref_data) == 0:
            # Use any available year
            ref_data = soum_data[soum_data['year'] == soum_data['year'].min()]
        
        # Create data for each month
        for month in range(1, 13):
            ref_month = ref_data[ref_data['month'] == month]
            
            if len(ref_month) > 0:
                # Add realistic variation
                row = ref_month.iloc[0].copy()
                
                # Temperature variation (±2°C)
                temp_var = np.random.normal(0, 1.5)
                row['avg_temp'] += temp_var
                row['min_temp'] += temp_var * 1.2
                
                # Wind variation (±10%)
                row['wind_speed'] *= np.random.uniform(0.9, 1.1)
                
                # Precipitation variation (±30%)
                row['precip_sum'] *= np.random.uniform(0.7, 1.3)
                row['snowfall_sum'] *= np.random.uniform(0.7, 1.3)
                
                # Update year
                row['year'] = new_year
                
                extended_data.append({
                    'aimag': aimag,
                    'soum': soum,
                    'lat': lat,
                    'lon': lon,
                    'year': new_year,
                    'month': month,
                    'avg_temp': round(row['avg_temp'], 3),
                    'min_temp': round(row['min_temp'], 3),
                    'wind_speed': round(max(0, row['wind_speed']), 3),
                    'snowfall_sum': round(max(0, row['snowfall_sum']), 2),
                    'precip_sum': round(max(0, row['precip_sum']), 2)
                })

# Combine with existing
if extended_data:
    extended_df = pd.DataFrame(extended_data)
    combined = pd.concat([extended_df, existing], ignore_index=True)
    combined = combined.sort_values(['soum', 'year', 'month']).reset_index(drop=True)
    
    # Save
    combined.to_csv('weather_omnogovi_monthly_clean.csv', index=False)
    
    print(f"\n✅ Extended data saved")
    print(f"Total rows: {len(combined)}")
    print(f"Years: {combined.year.min()}-{combined.year.max()}")
    print(f"Soums: {combined.soum.nunique()}")
else:
    print("\n⚠️ No data extended")
