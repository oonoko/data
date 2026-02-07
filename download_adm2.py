import json, urllib.request

with open("gb_mng_adm2_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

# geoBoundaries API нь downloadURL/ gjDownloadURL зэрэг талбар өгдөг (хувилбараас шалтгаалж нэр өөр байж болно)
url = meta.get("gjDownloadURL") or meta.get("downloadURL") or meta.get("gjDownloadUrl") or meta.get("downloadUrl")
if not url:
    raise SystemExit("GeoJSON download URL олдсонгүй. meta keys: " + ", ".join(meta.keys()))

print("Downloading:", url)
urllib.request.urlretrieve(url, "MNG_ADM2.geojson")
print("Saved: MNG_ADM2.geojson")
