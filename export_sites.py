"""Zapisuje sites.json dla index.html i pobiera profile wysokości do profiles/<ciało>/<id>.png.
Dostosuj import i funkcję fetch_profile do swojej biblioteki."""
import json
import os

DEFAULT_BODY = "mars"      # używane, gdy krotka nie ma 8. elementu
PROFILE_DIR = "profiles"   # profiles/mars/1.png, profiles/moon/2.png ... (id mogą się powtarzać między ciałami)

# from twoja_biblioteka import SITES, fetch_profile
# Krotka: (id, nazwa, szer, dł, nazwa_ziemia, szer_ziemia, dł_ziemia[, "mars" | "moon"])
#  - brak odpowiednika na Ziemi: nazwa_ziemia, szer_ziemia, dł_ziemia = None
#  - brak nazwy (None): na stronie pokaże się id (czyli nazwa pliku [id].png)
SITES = [
    (1, "Starship proposed landing site", 39.8000, -157.3900,
        "Mars Desert Research Station", 38.4053, -110.7900, "mars"),                 # Mars (domyślnie)
    (2, "Tranquility Base", 0.6741, 23.4729,
        "Mauna Kea", 19.8207, -155.4681, "moon"),                           # Księżyc (przykład)
    (3, None, -5.4, 137.8, None, None, None),                               # Mars, bez odpowiednika na Ziemi (przykład)
]


def fetch_profile(site_id, body, lat, lon, out_path):
    """TU PODEPNIJ SWOJĄ FUNKCJĘ: pobiera profil wysokości w okolicy (lat, lon)
    i zapisuje go jako PNG w out_path. Dopóki jej nie podłączysz, żaden punkt nie dostanie zdjęcia."""
    raise NotImplementedError


def get_profile(site_id, body, lat, lon):
    """Zwraca ścieżkę względną do PNG albo None, jeśli zdjęcia nie ma."""
    out_path = os.path.join(PROFILE_DIR, body, f"{site_id}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not os.path.exists(out_path):
        try:
            fetch_profile(site_id, body, lat, lon, out_path)
        except NotImplementedError:
            return None
        except Exception as err:                      # jedno nieudane pobranie nie przerywa reszty
            print(f"  ! profil dla {site_id} ({body}): {err}")
            return None
    return f"{PROFILE_DIR}/{body}/{site_id}.png" if os.path.exists(out_path) else None


def to_dict(t):
    id_, s_name, s_lat, s_lon, e_name, e_lat, e_lon, *rest = t
    body = (rest[0] if rest else DEFAULT_BODY).lower()
    d = {
        "id": id_,
        "body": body,
        "site": {"name": s_name or str(id_), "lat": s_lat, "lon": s_lon},
        "earth": None if e_lat is None or e_lon is None
                 else {"name": e_name or str(id_), "lat": e_lat, "lon": e_lon},
    }
    profile = get_profile(id_, body, s_lat, s_lon)
    if profile:
        d["profile"] = profile                         # bez tego pola punkt nie ma "Details..."
    return d


with open("sites.json", "w", encoding="utf-8") as f:
    json.dump([to_dict(s) for s in SITES], f, ensure_ascii=False, indent=2)
print(f"Zapisano {len(SITES)} lokalizacji do sites.json")
