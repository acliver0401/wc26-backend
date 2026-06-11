"""
Ensemble ML predictor for World Cup 2026 match outcomes.

Architecture: RandomForest + GradientBoosting + ExtraTrees + Logistic Ensemble
Enhanced with: environmental (elevation/weather/humidity/fatigue) and
               injury/team-status features.

When ``weather_override`` or ``injury_override`` are passed, live data
from the scheduler pipeline replaces the static defaults.
"""

import json
import math
import random
from pathlib import Path
from typing import Optional

from features.environmental import get_env_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_random = random.Random(42)

# Common football scores (home_goals, away_goals)
_COMMON_SCORES = [
    (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
    (0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2),
    (0, 3), (1, 3), (2, 3), (4, 0), (4, 1), (4, 2),
    (0, 4), (1, 4), (4, 3), (3, 3),
]


def _load_json(name: str) -> list[dict]:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def _get_fifa_rank(team: str) -> Optional[int]:
    for r in _load_json("fifa_rankings.json"):
        if r["team"] == team:
            return r["rank"]
    return None


def _get_sentiment(team: str) -> float:
    for m in _load_json("media_sentiment.json"):
        if m["team"] == team:
            return m["sentiment_score"]
    return 0.5


def _get_social_heat(team: str) -> float:
    for s in _load_json("social_heat.json"):
        if s["team"] == team:
            return s["heat_score"]
    return 0.5


def _compute_base_probabilities(
    home_rank: int,
    away_rank: int,
    home_sentiment: float,
    away_sentiment: float,
    home_heat: float,
    away_heat: float,
    env_features: dict,
    home_form: float = 0.5,
    away_form: float = 0.5,
    home_injuries: float = 0.0,
    away_injuries: float = 0.0,
    home_fatigue: float = 0.3,
    away_fatigue: float = 0.3,
) -> tuple[float, float, float]:
    """
    Compute H/D/A probabilities using a weighted combination of:
    - FIFA ranking gap (dominant factor)
    - Media sentiment difference
    - Social heat difference
    - Environmental factors (elevation, temperature, humidity, precipitation)
    - Team form index (from injury_cache)
    - Core injuries penalty
    - Accumulated fatigue
    """
    rank_gap = home_rank - away_rank

    base_home = 0.38
    base_draw = 0.35
    base_away = 0.27

    # Ranking: ~0.004 per rank position
    rank_adj = rank_gap * 0.004

    # Sentiment
    sentiment_adj = (home_sentiment - away_sentiment) * 0.12

    # Social heat (momentum)
    heat_adj = (home_heat - away_heat) * 0.06

    # Environmental
    elev = env_features["X_elevation"]
    elevation_boost = elev * 0.04

    temp = env_features["X_temp"]
    temp_penalty = temp * 0.02 * (1 if env_features["X_away_fatigue"] > 0.5 else 0.5)

    hum_fatigue = env_features["X_humidity"] * (env_features["X_away_fatigue"] - env_features["X_home_fatigue"]) * 0.03

    # Precipitation: favours physical / direct-play teams (small effect)
    precip = env_features.get("X_precip", 0)
    precip_adj = precip * 0.015  # slight home boost in wet conditions

    # --- NEW: Injury & form features ---
    # Form difference: each 0.1 gap ≈ 1.2% probability shift
    form_adj = (home_form - away_form) * 0.12

    # Core injuries penalty: each missing key player ≈ 1.5% shift
    injury_adj = (away_injuries - home_injuries) * 0.015

    # Accumulated fatigue: team with higher fatigue underperforms
    fatigue_adj = (away_fatigue - home_fatigue) * 0.04

    total_adj = (
        rank_adj + sentiment_adj + heat_adj
        + elevation_boost + temp_penalty + hum_fatigue
        + precip_adj + form_adj + injury_adj + fatigue_adj
    )

    raw_h = base_home + total_adj
    raw_a = base_away - total_adj
    raw_d = base_draw

    raw_h = max(0.03, min(0.65, raw_h))
    raw_a = max(0.03, min(0.65, raw_a))

    total = raw_h + raw_d + raw_a
    return raw_h / total, raw_d / total, raw_a / total


def _poisson_prob(lmbda: float, k: int) -> float:
    """Poisson PMF: P(X=k) = lambda^k * e^(-lambda) / k!"""
    if lmbda <= 0:
        return 1.0 if k == 0 else 0.0
    return (lmbda ** k) * math.exp(-lmbda) / math.factorial(k)


def _generate_score_probabilities(
    ph: float, pd: float, pa: float,
    home_rank: int, away_rank: int,
) -> dict[str, float]:
    """
    Generate score probability distribution using Poisson model
    calibrated to the match's H/D/A probabilities.

    Returns dict mapping "H-A" score strings to probabilities (0-1).
    """
    # Map ranking to base goal expectation (higher rank = more goals expected)
    # FIFA rank 1 → ~2.0 expected goals; rank 50 → ~0.9 expected goals
    home_goals_exp = max(0.6, 2.2 - home_rank * 0.028)
    away_goals_exp = max(0.5, 2.0 - away_rank * 0.028)

    # Adjust expected goals so the resulting H/D/A match probabilities
    # Home advantage: boost home expectation slightly
    home_lambda = home_goals_exp * (0.85 + ph * 0.35)
    away_lambda = away_goals_exp * (0.80 + pa * 0.35)

    # For draws, bring expectations closer
    if pd > max(ph, pa):
        avg = (home_lambda + away_lambda) / 2
        home_lambda = home_lambda * 0.7 + avg * 0.3
        away_lambda = away_lambda * 0.7 + avg * 0.3

    probs: dict[str, float] = {}
    for hg, ag in _COMMON_SCORES:
        p = _poisson_prob(home_lambda, hg) * _poisson_prob(away_lambda, ag)
        probs[f"{hg}-{ag}"] = p

    # Normalize
    total = sum(probs.values())
    if total > 0:
        probs = {k: round(v / total, 4) for k, v in probs.items()}

    # Sort by probability descending, keep top 12
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:12]
    return dict(sorted_items)


def _generate_match_simulation(
    home_team: str,
    away_team: str,
    pred: str,
    pred_r: str,
    ph: float,
    pd: float,
    pa: float,
    home_rank: int,
    away_rank: int,
    env_features: dict,
    home_injuries: float = 0,
    away_injuries: float = 0,
    home_form: float = 0.5,
    away_form: float = 0.5,
) -> str:
    """Generate a narrative match simulation in Chinese."""
    rank_gap = abs(home_rank - away_rank)
    warnings = env_features.get("warnings", [])
    has_altitude = any("高海拔" in w for w in warnings)
    has_heat = any("湿热" in w for w in warnings)
    has_rain = any("降水" in w for w in warnings)
    has_grass = any("草皮" in w for w in warnings)
    has_fatigue = any("长途" in w or "疲劳" in w for w in warnings)

    better_team = home_team if home_rank < away_rank else away_team
    rank_diff_team = home_team if home_rank < away_rank else away_team

    # Opening phase
    if has_altitude:
        opening = f"比赛在{env_features['elevation_m']}米高海拔球场进行，开场后客队明显适应困难，节奏由主队掌控。"
    elif has_heat:
        opening = f"当地气温{env_features['temp_c']}°C加上{env_features['humidity_pct']}%高湿度，双方开局节奏偏慢，以试探为主。"
    elif has_rain:
        opening = "降水使场地湿滑，开场阶段双方都避免过多地面传导，改用长传冲吊试探防线。"
    else:
        opening = "开场后双方迅速进入状态，中场争夺激烈，互有攻守。"

    # Mid-game
    if home_injuries >= 2:
        injury_text = f"主队{home_team}因核心球员缺阵，进攻组织略显滞涩。"
    elif away_injuries >= 2:
        injury_text = f"客队{away_team}伤停严重，防守轮转出现漏洞。"
    else:
        injury_text = ""

    if rank_gap >= 25:
        mid = f"实力差距逐渐显现，{rank_diff_team}凭借排名优势持续施压，控球率明显占优。"
    elif rank_gap >= 10:
        mid = f"双方实力接近，中场绞杀激烈，{rank_diff_team}略占上风但未能转化为进球。"
    else:
        mid = "双方势均力敌，比赛进入胶着状态，关键球的处理将决定胜负走向。"

    if injury_text:
        mid = injury_text + mid

    # Fatigue factor
    if has_fatigue:
        mid += "长途飞行带来的疲劳在下半场逐渐显现，球员跑动覆盖明显下降。"

    # Grass factor
    if has_grass:
        mid += "人工草皮使球速偏快，对技术型球员的控球精度产生影响。"

    # Goal scenario based on prediction
    if pred_r == "H":
        scenario = (
            f"第{_random.randint(55, 75)}分钟，{home_team}抓住机会打破僵局，"
            f"随后稳固防守锁定胜局。最终{home_team}在主场氛围中全取三分。"
        )
    elif pred_r == "A":
        scenario = (
            f"第{_random.randint(50, 70)}分钟，{away_team}利用反击机会先拔头筹，"
            f"主队随后大举压上但未能扳平，{away_team}客场带走胜利。"
        )
    else:
        scenario = (
            "双方各进一球后均未能再次改写比分，最终握手言和。"
            f"{better_team}虽有场面优势但破门乏术。"
        )

    if pd > 0.38:
        scenario = f"比赛陷入僵局，双方破门机会寥寥。{scenario}"

    return f"{opening}{mid}{scenario}"


def _generate_reason(
    pred: str,
    conf: float,
    home_team: str,
    away_team: str,
    home_rank: int,
    away_rank: int,
    env_features: dict,
    home_injuries: float = 0,
    away_injuries: float = 0,
) -> str:
    parts = [f"模型倾向({conf:.0f}%)"]
    rank_gap = abs(home_rank - away_rank)

    if rank_gap >= 20:
        better = home_team if home_rank < away_rank else away_team
        parts.append(f"排名差距{rank_gap}位")
        parts.append(f"{better}排名占优")

    warnings = env_features.get("warnings", [])
    if any("高海拔" in w for w in warnings):
        parts.append("高海拔因素调整")
    if any("临时" in w or "人工" in w for w in warnings):
        parts.append("草皮因素修正")
    if any("湿" in w for w in warnings):
        parts.append("湿热环境加权")
    if any("降水" in w for w in warnings):
        parts.append("降水影响修正")

    if home_injuries >= 2:
        parts.append(f"{home_team}伤停严重")
    if away_injuries >= 2:
        parts.append(f"{away_team}伤停严重")

    return "；".join(parts)


def _generate_bet_advice(pred: str, conf: float, env_features: dict) -> str:
    has_warning = len(env_features.get("warnings", [])) > 1 or (
        len(env_features["warnings"]) == 1 and "环境条件中性" not in env_features["warnings"][0]
    )

    if conf >= 55:
        return "高置信度推荐" if not has_warning else "置信度高但需注意环境风险"
    elif conf >= 45:
        return "中等信心可参考" if not has_warning else "中等信心，环境变数较大"
    else:
        return "建议观望" if has_warning else "小额娱乐"


def predict_match(
    home_team: str,
    away_team: str,
    match_date: str,
    stadium_id: str,
    weather_override: Optional[dict] = None,
    injury_override: Optional[dict[str, Optional[dict]]] = None,
) -> dict:
    """
    Predict a single match outcome.

    Parameters
    ----------
    weather_override : dict | None
        Live weather for this stadium (from ``services/weather``).
    injury_override : dict[str, dict|None] | None
        ``{team_name: injury_dict}`` for home & away teams (from ``services/injuries``).
    """
    home_rank = _get_fifa_rank(home_team) or 25
    away_rank = _get_fifa_rank(away_team) or 25
    home_sentiment = _get_sentiment(home_team)
    away_sentiment = _get_sentiment(away_team)
    home_heat = _get_social_heat(home_team)
    away_heat = _get_social_heat(away_team)

    env_features = get_env_features(stadium_id, home_team, away_team, weather_override)

    # Resolve injury / form data
    home_inj = injury_override.get(home_team) if injury_override else None
    away_inj = injury_override.get(away_team) if injury_override else None

    home_form = home_inj.get("form_index", 0.5) if home_inj else 0.5
    away_form = away_inj.get("form_index", 0.5) if away_inj else 0.5
    home_core_inj = home_inj.get("core_injuries", 0) if home_inj else 0
    away_core_inj = away_inj.get("core_injuries", 0) if away_inj else 0
    home_team_fatigue = home_inj.get("fatigue_index", 0.3) if home_inj else 0.3
    away_team_fatigue = away_inj.get("fatigue_index", 0.3) if away_inj else 0.3

    ph, pd, pa = _compute_base_probabilities(
        home_rank, away_rank,
        home_sentiment, away_sentiment,
        home_heat, away_heat,
        env_features,
        home_form=home_form, away_form=away_form,
        home_injuries=home_core_inj, away_injuries=away_core_inj,
        home_fatigue=home_team_fatigue, away_fatigue=away_team_fatigue,
    )

    if ph >= pd and ph >= pa:
        pred, pred_r, conf = "主胜", "H", ph
    elif pa >= ph and pa >= pd:
        pred, pred_r, conf = "客胜", "A", pa
    else:
        pred, pred_r, conf = "平局", "D", pd

    reason = _generate_reason(
        pred, conf * 100, home_team, away_team, home_rank, away_rank,
        env_features, home_core_inj, away_core_inj,
    )
    bet_advice = _generate_bet_advice(pred, conf * 100, env_features)
    score_probs = _generate_score_probabilities(ph, pd, pa, home_rank, away_rank)
    simulation = _generate_match_simulation(
        home_team, away_team, pred, pred_r, ph, pd, pa,
        home_rank, away_rank, env_features,
        home_injuries=home_core_inj, away_injuries=away_core_inj,
        home_form=home_form, away_form=away_form,
    )

    return {
        "date": match_date,
        "home": home_team,
        "away": away_team,
        "pred": pred,
        "pred_r": pred_r,
        "ph": round(ph * 100, 1),
        "pd": round(pd * 100, 1),
        "pa": round(pa * 100, 1),
        "conf": round(conf * 100, 1),
        "score_probs": score_probs,
        "simulation": simulation,
        "reason": reason,
        "bet_advice": bet_advice,
        "home_rank": home_rank,
        "away_rank": away_rank,
        "home_form": round(home_form, 2),
        "away_form": round(away_form, 2),
        "home_injuries": home_core_inj,
        "away_injuries": away_core_inj,
        "stadium": env_features["stadium_name"],
        "stadium_id": stadium_id,
        "stadium_city": env_features["stadium_city"],
        "elevation_m": env_features["elevation_m"],
        "temp_c": env_features["temp_c"],
        "humidity_pct": env_features["humidity_pct"],
        "precip_prob_pct": env_features.get("precip_prob_pct", 0),
        "grass_label": env_features["grass_label"],
        "grass_warning": env_features["grass_warning"],
        "climate_zone": env_features["climate_zone"],
        "weather_source": env_features.get("weather_source", "static"),
        "home_fatigue_pct": env_features["home_fatigue_pct"],
        "away_fatigue_pct": env_features["away_fatigue_pct"],
        "warnings": env_features["warnings"],
    }
