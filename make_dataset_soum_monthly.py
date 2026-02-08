#!/usr/bin/env python3
"""
Сум+сар dataset үүсгэх скрипт
Weather monthly + livestock synthetic → final dataset
"""

import pandas as pd
import numpy as np

# 1. Weather monthly clean уншиx
print("Loading weather data...")
weather = pd.read_csv('weather_omnogovi_monthly_clean.csv')
print(f"Weather data: {len(weather)} rows")

# 2. Livestock yearly уншиx
print("Loading livestock data...")
livestock_yearly = pd.read_csv('livestock_total_yearly.csv')
print(f"Livestock yearly: {len(livestock_yearly)} rows")

# 3. Сумын жагсаалт
soums = weather[['aimag', 'soum', 'lat', 'lon']].drop_duplicates()
n_soums = len(soums)
print(f"Number of soums: {n_soums}")

# 4. Малын тоог сум бүрт хуваарилах (Dirichlet random weight)
np.random.seed(42)  # reproducibility

livestock_soum_month = []

for _, row in livestock_yearly.iterrows():
    year = int(row['year'])
    total = row['total_livestock']
    
    # Dirichlet weight (сум бүрт санамсаргүй хувь)
    weights = np.random.dirichlet(np.ones(n_soums))
    soum_totals = weights * total
    
    # 12 сар руу тараах (жилийн дундаж)
    for i, (_, soum_row) in enumerate(soums.iterrows()):
        soum_name = soum_row['soum']
        soum_total = soum_totals[i]
        
        # Сар бүрт жижиг хэлбэлзэл нэмэх (±5%)
        for month in range(1, 13):
            variation = np.random.uniform(0.95, 1.05)
            livestock_month = soum_total * variation
            
            livestock_soum_month.append({
                'aimag': soum_row['aimag'],
                'soum': soum_name,
                'year': year,
                'month': month,
                'livestock_count': round(livestock_month, 1)
            })

livestock_df = pd.DataFrame(livestock_soum_month)
print(f"Livestock soum-month: {len(livestock_df)} rows")

# 5. Weather + Livestock merge
print("Merging weather and livestock...")
final_df = weather.merge(
    livestock_df,
    on=['aimag', 'soum', 'year', 'month'],
    how='left'
)

# 6. Risk score тооцоолох (0-100)
print("Calculating risk scores...")

def calculate_risk_score(row):
    """
    Risk score based on weather + livestock exposure
    """
    score = 0
    
    # Temperature risk (min_temp буурах тусам risk↑)
    if row['min_temp'] < -20:
        score += 30
    elif row['min_temp'] < -15:
        score += 20
    elif row['min_temp'] < -10:
        score += 10
    
    # Wind risk (wind_speed өсөх тусам risk↑)
    if row['wind_speed'] > 15:
        score += 20
    elif row['wind_speed'] > 12:
        score += 10
    
    # Snowfall risk (snowfall_sum өсөх тусам risk↑)
    if row['snowfall_sum'] > 5:
        score += 20
    elif row['snowfall_sum'] > 2:
        score += 10
    
    # Precipitation risk (хэт бага үед хуурайшилт)
    if row['precip_sum'] < 5:
        score += 15
    elif row['precip_sum'] < 10:
        score += 5
    
    # Livestock exposure (мал их байх тусам exposure↑)
    if pd.notna(row['livestock_count']):
        if row['livestock_count'] > 100:
            score += 15
        elif row['livestock_count'] > 50:
            score += 10
        elif row['livestock_count'] > 20:
            score += 5
    
    return min(score, 100)  # cap at 100

final_df['risk_score'] = final_df.apply(calculate_risk_score, axis=1)

# 7. Risk level (0-3) and label
def get_risk_level(score):
    if score < 25:
        return 0, 'Бага'
    elif score < 50:
        return 1, 'Дунд'
    elif score < 75:
        return 2, 'Өндөр'
    else:
        return 3, 'Маш өндөр'

final_df[['risk_level', 'risk_label']] = final_df['risk_score'].apply(
    lambda x: pd.Series(get_risk_level(x))
)

# 8. Баганын дараалал
columns_order = [
    'aimag', 'soum', 'lat', 'lon', 'year', 'month',
    'avg_temp', 'min_temp', 'wind_speed', 'snowfall_sum', 'precip_sum',
    'livestock_count', 'risk_score', 'risk_level', 'risk_label'
]

final_df = final_df[columns_order]

# 9. Save
output_file = 'final_dzud_ai_dataset_soum_monthly.csv'
final_df.to_csv(output_file, index=False)
print(f"\n✅ Dataset saved: {output_file}")
print(f"Total rows: {len(final_df)}")
print(f"\nRisk distribution:")
print(final_df['risk_label'].value_counts())
print(f"\nSample data:")
print(final_df.head(10))
