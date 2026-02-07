import geopandas as gpd
import pandas as pd
import unicodedata

adm2 = gpd.read_file("MNG_ADM2.geojson").to_crs(epsg=4326)
adm1 = gpd.read_file("MNG_ADM1.geojson").to_crs(epsg=4326)

joined = gpd.sjoin(adm2, adm1, how="left", predicate="within")

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# join-ийн дараа нэрүүд suffix-тэй болж өөрчлөгдөж болно
soum_col = pick_col(joined, ["shapeName_left", "shapeName", "NAME_2", "ADM2_NAME", "adm2_name"])
aimag_col = pick_col(joined, ["shapeName_right", "shapeName", "NAME_1", "ADM1_NAME", "adm1_name"])

if not soum_col or not aimag_col:
    raise SystemExit(f"Name columns олдсонгүй. columns={list(joined.columns)}")

def strip_accents(s: str) -> str:
    # Ömnögovi -> Omnogovi
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

aimag_series = joined[aimag_col].astype(str).str.strip().fillna("")
aimag_norm = aimag_series.map(strip_accents).str.lower()

# яг “omnogovi” гэж шүүнэ
g_om = joined[aimag_norm == "omnogovi"].copy()

if len(g_om) == 0:
    # fallback: contains
    g_om = joined[aimag_norm.str.contains("omnogovi", na=False)].copy()

if len(g_om) == 0:
    uniq = sorted(set(aimag_series.unique().tolist()))
    raise SystemExit("Өмнөговь олдсонгүй. ADM1 нэрс (жишээ 30): " + ", ".join(uniq[:30]))

# centroid-ийг projected CRS дээр зөв гаргана (UTM 48N)
g_om_utm = g_om.to_crs(epsg=32648)
centroids = g_om_utm.geometry.centroid
centroids = gpd.GeoSeries(centroids, crs="EPSG:32648").to_crs(epsg=4326)

g_om["lat"] = centroids.y
g_om["lon"] = centroids.x

out = pd.DataFrame({
    "aimag": g_om[aimag_col].astype(str),
    "soum": g_om[soum_col].astype(str),
    "lat": g_om["lat"].round(6),
    "lon": g_om["lon"].round(6),
}).drop_duplicates().sort_values(["aimag", "soum"])

out.to_csv("soum_list.csv", index=False, encoding="utf-8")
print("✅ Saved soum_list.csv")
print(out.head(20))
print("Total soums:", len(out))
