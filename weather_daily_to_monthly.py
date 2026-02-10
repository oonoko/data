#!/usr/bin/env python3
"""
Convert daily weather data to monthly aggregates
"""

import pandas as pd

df = pd.read_csv("weather_omnogovi_daily.csv")

# time -> datetime
df["time"] = pd.to_datetime(df["time"])
df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month

# Monthly aggregate
monthly = (
    df.groupby(["aimag", "soum", "lat", "lon", "year", "month"], as_index=False)
      .agg(
          avg_temp=("temperature_2m_mean", "mean"),
          min_temp=("temperature_2m_min", "min"),
          wind_speed=("wind_speed_10m_mean", "mean"),
          snowfall_sum=("snowfall_sum", "sum"),
          precip_sum=("precipitation_sum", "sum"),
      )
)

# round for cleanliness
for c in ["avg_temp","min_temp","wind_speed","snowfall_sum","precip_sum"]:
    monthly[c] = monthly[c].round(3)

monthly.to_csv("weather_omnogovi_monthly.csv", index=False, encoding="utf-8")
print("✅ Saved weather_omnogovi_monthly.csv", monthly.shape)
print(monthly.head(10))
