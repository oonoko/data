import json, urllib.request

with open("gb_mng_adm1_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

url = meta.get("gjDownloadURL") or meta.get("downloadURL") or meta.get("gjDownloadUrl") or meta.get("downloadUrl")
if not url:
    raise SystemExit("ADM1 GeoJSON download URL олдсонгүй. meta keys: " + ", ".join(meta.keys()))

print("Downloading:", url)
urllib.request.urlretrieve(url, "MNG_ADM1.geojson")
print("Saved: MNG_ADM1.geojson")
