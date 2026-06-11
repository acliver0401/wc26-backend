"""
Environmental feature engineering for World Cup 2026 stadiums.

Adds feature dimensions to the prediction X matrix:
  - stadium_elevation: raw elevation in meters (normalized)
  - match_temp: game-day temperature (Celsius, normalized) — from live weather if available
  - match_humidity: game-day humidity (%, normalized) — from live weather if available
  - flight_fatigue: travel distance impact score (0-1, normalized)
  - precip_risk: precipitation probability affecting play style (0-1)

Plus derived composite risk indicators used for UI warnings.

When ``weather_override`` is provided (from the scheduler's Open-Meteo fetch),
live temperature / humidity / precipitation values replace the static June
averages from stadium_meta.json.
"""

import json
import math
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_stadiums() -> list[dict]:
    with open(DATA_DIR / "stadium_meta.json", encoding="utf-8") as f:
        return json.load(f)


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


TEAM_HOME_COORDS: dict[str, tuple[float, float]] = {
    "Argentina": (-34.6037, -58.3816), "Brazil": (-15.7975, -47.8919),
    "Uruguay": (-34.9011, -56.1645), "Colombia": (4.7110, -74.0721),
    "Ecuador": (-0.1807, -78.4678), "Paraguay": (-25.2637, -57.5759),
    "France": (48.8566, 2.3522), "Spain": (40.4168, -3.7038),
    "Germany": (52.5200, 13.4050), "England": (51.5074, -0.1278),
    "Portugal": (38.7223, -9.1393), "Netherlands": (52.3676, 4.9041),
    "Belgium": (50.8503, 4.3517), "Croatia": (45.8150, 15.9819),
    "Switzerland": (46.9480, 7.4474), "Austria": (48.2082, 16.3738),
    "Norway": (59.9139, 10.7522), "Sweden": (59.3293, 18.0686),
    "Scotland": (55.9533, -3.1883), "Czechia": (50.0755, 14.4378),
    "Turkey": (39.9334, 32.8597), "Bosnia-Herzegovina": (43.8563, 18.4131),
    "Mexico": (19.4326, -99.1332), "United States": (38.9072, -77.0369),
    "Canada": (45.4215, -75.6972), "Japan": (35.6762, 139.6503),
    "South Korea": (37.5665, 126.9780), "Australia": (-35.2809, 149.1300),
    "Saudi Arabia": (24.7136, 46.6753), "Iran": (35.6892, 51.3890),
    "Iraq": (33.3152, 44.3661), "Jordan": (31.9539, 35.9106),
    "Uzbekistan": (41.2995, 69.2401), "Qatar": (25.2854, 51.5310),
    "Morocco": (34.0209, -6.8416), "Senegal": (14.7167, -17.4677),
    "Egypt": (30.0444, 31.2357), "Algeria": (36.7538, 3.0588),
    "Tunisia": (36.8065, 10.1815), "Ghana": (5.6037, -0.1870),
    "Ivory Coast": (5.3600, -4.0083), "Congo DR": (-4.3258, 15.3132),
    "South Africa": (-25.7479, 28.2293),
    "Panama": (8.9824, -79.5199), "Haiti": (18.5944, -72.3074),
    "New Zealand": (-41.2865, 174.7762),
    "Cape Verde Islands": (14.9330, -23.5133), "Curaçao": (12.1696, -68.9900),
}


def compute_flight_fatigue(team: str, stadium_lat: float, stadium_lng: float) -> float:
    coords = TEAM_HOME_COORDS.get(team)
    if coords is None:
        return 0.5
    dist = _haversine_km(coords[0], coords[1], stadium_lat, stadium_lng)
    if dist < 500:
        return 0.0
    if dist > 8000:
        return 1.0
    return (dist - 500) / 7500.0


def get_env_features(
    stadium_id: str,
    home_team: str,
    away_team: str,
    weather_override: Optional[dict] = None,
) -> dict:
    """
    Compute the full environmental feature vector for a match.

    Parameters
    ----------
    weather_override : dict | None
        Live weather from Open-Meteo (see ``services/weather.py``).
        When provided its ``temp_max_c`` / ``humidity_max_pct`` /
        ``precip_prob_max_pct`` replace the static stadium averages.
    """
    stadiums = _load_stadiums()
    stadium = next((s for s in stadiums if s["id"] == stadium_id), None)
    if stadium is None:
        raise ValueError(f"Unknown stadium_id: {stadium_id}")

    # --- Resolve temperature & humidity (live > static) ---
    if weather_override and isinstance(weather_override, dict):
        w = weather_override
        temp_c = w.get("temp_max_c") if w.get("temp_max_c") is not None else stadium["avg_temp_june_c"]
        humidity_pct = w.get("humidity_max_pct") if w.get("humidity_max_pct") is not None else stadium["avg_humidity_june_pct"]
        precip_prob = w.get("precip_prob_max_pct", 0) or 0
    else:
        temp_c = stadium["avg_temp_june_c"]
        humidity_pct = stadium["avg_humidity_june_pct"]
        precip_prob = 0

    # Normalize
    elevation_norm = min(stadium["elevation_m"] / 2500.0, 1.0)
    temp_norm = (temp_c - 15) / 25.0  # 15-40°C range
    humidity_norm = humidity_pct / 100.0
    precip_norm = precip_prob / 100.0

    home_fatigue = compute_flight_fatigue(home_team, stadium["coordinates"]["lat"], stadium["coordinates"]["lng"])
    away_fatigue = compute_flight_fatigue(away_team, stadium["coordinates"]["lat"], stadium["coordinates"]["lng"])

    warnings = _generate_warnings(stadium, elevation_norm, temp_norm, humidity_norm, precip_norm, home_fatigue, away_fatigue)

    return {
        "X_elevation": round(elevation_norm, 4),
        "X_temp": round(temp_norm, 4),
        "X_humidity": round(humidity_norm, 4),
        "X_precip": round(precip_norm, 4),
        "X_home_fatigue": round(home_fatigue, 4),
        "X_away_fatigue": round(away_fatigue, 4),

        "stadium_name": stadium["name"],
        "stadium_city": stadium["city"],
        "elevation_m": stadium["elevation_m"],
        "temp_c": temp_c,
        "humidity_pct": humidity_pct,
        "precip_prob_pct": precip_prob,
        "grass_label": stadium["grass_label"],
        "grass_warning": stadium["grass_warning"],
        "climate_zone": stadium["climate_zone"],
        "home_fatigue_pct": round(home_fatigue * 100, 0),
        "away_fatigue_pct": round(away_fatigue * 100, 0),
        "weather_source": "live" if weather_override else "static",
        "warnings": warnings,
    }


def _generate_warnings(
    stadium: dict,
    elevation_norm: float,
    temp_norm: float,
    humidity_norm: float,
    precip_norm: float,
    home_fatigue: float,
    away_fatigue: float,
) -> list[str]:
    warnings: list[str] = []

    if stadium["elevation_m"] >= 2000:
        warnings.append("高海拔体能考验 — 2200m+ 含氧量下降约20%，对平原球队极为不利")
    elif stadium["elevation_m"] >= 1000:
        warnings.append(f"中高海拔 ({stadium['elevation_m']}m) — 需关注客队适应能力")

    if stadium["grass_warning"]:
        wtype = "临时加铺天然草皮" if "temporary" in stadium.get("grass_type", "") else "人工草皮"
        warnings.append(f"{wtype} — 球速反弹异常，谨防技术型球队发挥失常")

    if humidity_norm >= 0.70 and temp_norm >= 0.60:
        warnings.append(f"湿热环境 (🌡️{stadium.get('avg_temp_june_c', '?')}°C 💧{stadium.get('avg_humidity_june_pct', '?')}%) — 体能消耗剧增，体能型球队占优")
    elif humidity_norm >= 0.70:
        warnings.append(f"高湿度 — 皮球飞行阻力增大，远射威胁降低")

    if precip_norm >= 0.50:
        warnings.append(f"高降水概率 ({int(precip_norm * 100)}%) — 场地湿滑，技术型球队发挥受限，定位球权重上升")

    if home_fatigue > 0.7 and away_fatigue > 0.7:
        warnings.append("双方均为跨洲长途飞行，疲劳度相当，比赛节奏可能偏慢")
    elif away_fatigue > 0.7:
        warnings.append("客队长途跨洲飞行 (>8000km)，疲劳度极高，谨防下半场崩盘")

    if not warnings:
        warnings.append("环境条件中性，无明显极端因子影响")

    return warnings
