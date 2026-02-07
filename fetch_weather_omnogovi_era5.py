from datetime import date
import pandas as pd
import requests
import time

soums = pd.read_csv("soum_list.csv")

# хугацаа (чи хүсвэл өөрчилж болно)
start_date = "2015-01-01"
end_date   = "2024-12-31"

# Open-Meteo historical (ERA5/ECMWF reanalysis)
BASE = "https://archive-api.open-meteo.com/v1/archive"

# Бидэнд хэрэгтэй daily хувьсагчид:
# temperature_2m_mean ~ avg_temp
# temperature_2m_min  ~ min_temp
# wind_speed_10m_mean ~ wind_speed
# snowfall_sum        ~ snow proxy (байхгүй бол precipitation_sum ашиглана)
DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_min",
    "wind_speed_10m_mean",
    "snowfall_sum",
    "precipitation_sum"
]

all_rows = []
log_lines = []

for _, r in soums.iterrows():
    soum = r["soum"]
    aimag = r["aimag"]
    lat = float(r["lat"])
    lon = float(r["lon"])

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Asia/Ulaanbaatar"
    }

    try:
        resp = requests.get(BASE, params=params, timeout=60)
        if resp.status_code != 200:
            log_lines.append(f"[ERR] {soum}: HTTP {resp.status_code}")
            continue

        js = resp.json()
        daily = js.get("daily", {})
        if not daily or "time" not in daily:
            log_lines.append(f"[WARN] {soum}: no daily data in response")
            continue

        df = pd.DataFrame(daily)
        df["aimag"] = aimag
        df["soum"] = soum
        df["lat"] = lat
        df["lon"] = lon

        all_rows.append(df)
        log_lines.append(f"[OK] {soum}: rows={len(df)}")

        time.sleep(0.2)

    except Exception as e:
        log_lines.append(f"[ERR] {soum}: {type(e).__name__} - {e}")

if all_rows:
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv("weather_omnogovi_daily.csv", index=False, encoding="utf-8")
    print("✅ Saved weather_omnogovi_daily.csv", out.shape)
    print(out.head())
else:
    print("❌ No weather data collected")

with open("weather_fetch.log", "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

print("📝 Saved weather_fetch.log")
