"""Zapisuje sites.json dla index.html.
Dostosuj import i nazwę funkcji/zmiennej do swojej biblioteki."""
import json

# from twoja_biblioteka import SITES          # <- lista krotek (lub funkcja zwracająca listę)
SITES = [
    (1, "Starship landing site", 39.8000, -157.3900,
        "Mars Desert Research Station", 38.4053, -110.7900),
]

def to_dict(t):
    id_, m_name, m_lat, m_lon, e_name, e_lat, e_lon = t
    return {
        "id": id_,
        "mars":  {"name": m_name, "lat": m_lat, "lon": m_lon},
        "earth": {"name": e_name, "lat": e_lat, "lon": e_lon},
    }

with open("sites.json", "w", encoding="utf-8") as f:
    json.dump([to_dict(s) for s in SITES], f, ensure_ascii=False, indent=2)
print(f"Zapisano {len(SITES)} lokalizacji do sites.json")
