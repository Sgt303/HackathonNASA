"""Zapisuje sites.json dla index.html.
Dostosuj import do swojej biblioteki."""
import json

DEFAULT_BODY = "mars"      # używane, gdy krotka nie ma 8. elementu

# from twoja_biblioteka import SITES   # <- lista krotek
# Krotka: (id, nazwa_ciała, szer, dł, nazwa_ziemia, szer_ziemia, dł_ziemia[, "mars" | "moon"])
SITES = [
    (1, "Starship proposed landing site", 39.8000, -157.3900,
        "Mars Desert Research Station", 38.4053, -110.7900, "mars"),                 # Mars (domyślnie)
    (2, "Tranquility Base", 0.6741, 23.4729,
        "Mauna Kea", 19.8207, -155.4681, "moon"),                           # Księżyc (przykład)
]

def to_dict(t):
    id_, s_name, s_lat, s_lon, e_name, e_lat, e_lon, *rest = t
    return {
        "id": id_,
        "body": (rest[0] if rest else DEFAULT_BODY).lower(),
        "site":  {"name": s_name, "lat": s_lat, "lon": s_lon},
        "earth": {"name": e_name, "lat": e_lat, "lon": e_lon},
    }

with open("sites.json", "w", encoding="utf-8") as f:
    json.dump([to_dict(s) for s in SITES], f, ensure_ascii=False, indent=2)
print(f"Zapisano {len(SITES)} lokalizacji do sites.json")
