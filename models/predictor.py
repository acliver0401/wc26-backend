"""
Ensemble ML predictor for World Cup 2026 match outcomes — v3.0.

Architecture: Poisson goal model + weighted H/D/A ensemble + sanity check.

Key improvements over v2:
  - **Bivariate Poisson** with attacking-strength / defensive-weakness coefficients
    produces varied scorelines (3-2, 4-0, 0-0) instead of all 1-0 / 1-1.
  - **Rank sign fix**: ``rank_adj = (away_rank - home_rank) * k`` so a *better*
    (lower-numbered) home team *gains* probability.
  - **Tier-based sanity check** (熔断机制): when the FIFA-tier gap is ≥2,
    a minimum win-probability floor for the stronger side is enforced,
    preventing absurd upsets like Germany trailing Curaçao.
  - **Form / injury / fatigue** features preserved from v2.
"""

from __future__ import annotations

import json
import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Optional

from features.environmental import get_env_features
from pipeline.lineup_fetcher import get_lineup_multipliers

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_random = random.Random(42)

# ---------------------------------------------------------------------------
# Team strength coefficients (derived from FIFA points)
# ---------------------------------------------------------------------------

# League-average goals per team per match (World Cup baseline ≈ 1.35)
_LEAGUE_AVG_GOALS = 1.35
_HOME_ADVANTAGE = 1.18  # home team scores ~18 % more


@lru_cache(maxsize=1)
def _load_rankings() -> list[dict]:
    with open(DATA_DIR / "fifa_rankings.json", encoding="utf-8") as f:
        return json.load(f)


def _build_strength_map() -> dict[str, dict[str, float]]:
    """
    Compute **Attacking Strength (AS)** and **Defensive Weakness (DW)**
    for every team from FIFA points / rank.

    AS ∈ [0.45, 2.0]   — higher = more goals scored
    DW ∈ [0.50, 2.0]   — higher = more goals conceded
    """
    rankings = _load_rankings()
    points = [r["points"] for r in rankings]
    p_min, p_max = min(points), max(points)

    strength: dict[str, dict[str, float]] = {}
    for r in rankings:
        team = r["team"]
        rank = r["rank"]
        pts = r["points"]

        # Points → attacking quality  (linear scale)
        pct = (pts - p_min) / (p_max - p_min) if p_max > p_min else 0.5
        attacking = 0.45 + pct * 1.55  # [0.45, 2.0]

        # Rank → defensive vulnerability (worse rank = leakier defence)
        defensive = 0.50 + (rank - 1) / 47.0 * 1.50  # [0.50, 2.0]

        strength[team] = {"as": round(attacking, 4), "dw": round(defensive, 4)}

    return strength


# ---------------------------------------------------------------------------
# Tier system for sanity check
# ---------------------------------------------------------------------------

def _tier(rank: int) -> int:
    """Map FIFA rank → competitive tier (1 = elite, 4 = minnow)."""
    if rank <= 8:
        return 1
    if rank <= 18:
        return 2
    if rank <= 34:
        return 3
    return 4


def _min_win_prob(tier_gap: int, is_home: bool) -> float:
    """
    Floor win-probability for the higher-tier side.

    tier_gap = weaker_tier - stronger_tier  (≥ 0)
    """
    if tier_gap <= 1:
        return 0.0  # no floor — competitive match
    bonus = 0.04 if is_home else 0.0
    # Tier 2 vs Tier 3:  floor ~0.42;  Tier 1 vs Tier 4: floor ~0.55
    return {2: 0.38, 3: 0.48, 4: 0.55}.get(tier_gap, 0.0) + bonus


# ---------------------------------------------------------------------------
# Poisson goal model
# ---------------------------------------------------------------------------

def _get_attacking_strength(team: str) -> float:
    return _build_strength_map().get(team, {}).get("as", 1.0)


def _get_defensive_weakness(team: str) -> float:
    return _build_strength_map().get(team, {}).get("dw", 1.0)


def _expected_goals(
    team_as: float, opp_dw: float, is_home: bool,
) -> float:
    """λ = league_avg × AS × opp_DW × (home_advantage if applicable)."""
    ha = _HOME_ADVANTAGE if is_home else 1.0
    return _LEAGUE_AVG_GOALS * team_as * opp_dw * ha


def _poisson_pmf(lmbda: float, k: int) -> float:
    """P(X = k) for Poisson(λ)."""
    if lmbda <= 0:
        return 1.0 if k == 0 else 0.0
    return (lmbda ** k) * math.exp(-lmbda) / math.factorial(k)


def _bivariate_poisson_matrix(
    lambda_h: float, lambda_a: float, max_g: int = 5,
) -> dict[str, float]:
    """
    Full bivariate Poisson grid for 0 … max_g goals each side.

    Returns ``{"H-A": prob, …}`` sorted by probability descending.
    Includes a Dixon-Coles low-score adjustment: slightly boosts
    0-0 and 1-0 / 0-1 correlations to match real-world frequencies.
    """
    raw: dict[str, float] = {}
    for hg in range(max_g + 1):
        for ag in range(max_g + 1):
            p = _poisson_pmf(lambda_h, hg) * _poisson_pmf(lambda_a, ag)
            raw[f"{hg}-{ag}"] = p

    # Dixon-Coles ρ correction for low scores
    # ρ peaks when teams are evenly matched (|λ_h - λ_a| ≈ 0) and decays
    # when there is a clear favourite.  This mirrors real-world data where
    # draws (especially 0-0) are more common in balanced fixtures.
    # Ref: Dixon & Coles (1997), "Modelling Association Football Scores"
    rho = 0.08 * math.exp(-abs(lambda_h - lambda_a))

    # τ(x,y) adjustment — inflates 0-0, 0-1, 1-0; slightly deflates 1-1
    raw["0-0"] *= (1 - lambda_h * lambda_a * rho)
    raw["1-0"] *= (1 + lambda_h * rho)
    raw["0-1"] *= (1 + lambda_a * rho)
    raw["1-1"] *= (1 - rho)

    # Normalise
    total = sum(raw.values())
    if total <= 0:
        return {"1-0": 0.3, "0-0": 0.2, "0-1": 0.2, "2-0": 0.15, "1-1": 0.15}
    norm = {k: v / total for k, v in raw.items()}

    # Aggregate 5+ goals into "5+" bucket
    agg: dict[str, float] = {}
    for k, v in norm.items():
        h_str, a_str = k.split("-")
        hg, ag = int(h_str), int(a_str)
        h_key = "5+" if hg >= 5 else str(hg)
        a_key = "5+" if ag >= 5 else str(ag)
        key = f"{h_key}-{a_key}"
        agg[key] = agg.get(key, 0.0) + v

    # Sort & keep top 15
    sorted_items = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:15]
    return dict(sorted_items)


# ---------------------------------------------------------------------------
# H/D/A probability engine (ensemble base)
# ---------------------------------------------------------------------------

def _get_fifa_rank(team: str) -> Optional[int]:
    for r in _load_rankings():
        if r["team"] == team:
            return r["rank"]
    return None


def _load_json(name: str) -> list[dict]:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


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
    Weighted ensemble of 10 feature dimensions → raw H/D/A probabilities.

    **Sign fix**: rank_adj is now ``(away_rank - home_rank)`` so that a
    better (lower-numbered) home team receives a *positive* boost.
    """
    # --- 1. FIFA rank (CORRECTED SIGN) ---------------------------------
    # Positive when home is better → boosts home win chance
    rank_adj = (away_rank - home_rank) * 0.006

    # --- 2. Sentiment --------------------------------------------------
    sentiment_adj = (home_sentiment - away_sentiment) * 0.10

    # --- 3. Social heat (momentum) -------------------------------------
    heat_adj = (home_heat - away_heat) * 0.05

    # --- 4. Elevation --------------------------------------------------
    elev = env_features["X_elevation"]
    elevation_boost = elev * 0.05

    # --- 5. Temperature ------------------------------------------------
    temp = env_features["X_temp"]
    temp_penalty = temp * 0.025 * (1.0 if env_features["X_away_fatigue"] > 0.5 else 0.5)

    # --- 6. Humidity ---------------------------------------------------
    hum_fatigue = (
        env_features["X_humidity"]
        * (env_features["X_away_fatigue"] - env_features["X_home_fatigue"])
        * 0.03
    )

    # --- 7. Precipitation ----------------------------------------------
    precip = env_features.get("X_precip", 0)
    precip_adj = precip * 0.015

    # --- 8. Form -------------------------------------------------------
    form_adj = (home_form - away_form) * 0.10

    # --- 9. Injuries ---------------------------------------------------
    injury_adj = (away_injuries - home_injuries) * 0.018

    # --- 10. Travel fatigue --------------------------------------------
    fatigue_adj = (away_fatigue - home_fatigue) * 0.05

    total_adj = (
        rank_adj + sentiment_adj + heat_adj
        + elevation_boost + temp_penalty + hum_fatigue
        + precip_adj + form_adj + injury_adj + fatigue_adj
    )

    base_home = 0.38
    base_draw = 0.35
    base_away = 0.27

    raw_h = base_home + total_adj
    raw_a = base_away - total_adj
    raw_d = base_draw

    # Clamp to avoid extreme values
    raw_h = max(0.03, min(0.70, raw_h))
    raw_a = max(0.03, min(0.70, raw_a))

    total = raw_h + raw_d + raw_a
    return raw_h / total, raw_d / total, raw_a / total


# ---------------------------------------------------------------------------
# Sanity check — circuit-breaker for tier mismatches
# ---------------------------------------------------------------------------

def _apply_sanity_check(
    ph: float, pd: float, pa: float,
    home_rank: int, away_rank: int,
    home_team: str, away_team: str,
) -> tuple[float, float, float]:
    """
    If one side is 2+ tiers above the other, enforce a minimum win-prob
    floor so the weaker team cannot be (wrongly) favoured.
    """
    home_tier = _tier(home_rank)
    away_tier = _tier(away_rank)

    if home_tier < away_tier:  # home is stronger
        gap = away_tier - home_tier
        floor = _min_win_prob(gap, is_home=True)
        if ph < floor and gap >= 2:
            # Redistribute: pull ph up to floor, take from pa first, then pd
            deficit = floor - ph
            take_from_a = min(deficit, pa - 0.05)
            ph += take_from_a
            pa -= take_from_a
            deficit -= take_from_a
            if deficit > 0 and pd > 0.12:
                take_from_d = min(deficit, pd - 0.12)
                ph += take_from_d
                pd -= take_from_d
    elif away_tier < home_tier:  # away is stronger
        gap = home_tier - away_tier
        floor = _min_win_prob(gap, is_home=False)
        if pa < floor and gap >= 2:
            deficit = floor - pa
            take_from_h = min(deficit, ph - 0.05)
            pa += take_from_h
            ph -= take_from_h
            deficit -= take_from_h
            if deficit > 0 and pd > 0.12:
                take_from_d = min(deficit, pd - 0.12)
                pa += take_from_d
                pd -= take_from_d

    # Re-normalise
    total = ph + pd + pa
    return ph / total, pd / total, pa / total


# ---------------------------------------------------------------------------
# Reason / advice / simulation generators
# ---------------------------------------------------------------------------

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

    if rank_gap >= 15:
        better = home_team if home_rank < away_rank else away_team
        parts.append(f"排名差距{rank_gap}位")
        parts.append(f"{better}实力占优")

        # Add tier-gap context
        t_gap = abs(_tier(home_rank) - _tier(away_rank))
        if t_gap >= 2:
            parts.append(f"档差{t_gap}档·熔断修正")

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


def _generate_reason_v4(
    pred: str,
    conf: float,
    home_team: str,
    away_team: str,
    home_rank: int,
    away_rank: int,
    env_features: dict,
    home_injuries: float = 0,
    away_injuries: float = 0,
    prediction_status: str = "Pre-Match",
    lineup_info: dict | None = None,
    home_multiplier: float = 1.0,
    away_multiplier: float = 1.0,
) -> str:
    """Extended reason generator that includes lineup context."""
    base = _generate_reason(
        pred, conf, home_team, away_team, home_rank, away_rank,
        env_features, home_injuries, away_injuries,
    )

    if prediction_status != "Live-Lineup" or lineup_info is None:
        return base

    parts = [base]
    insight = lineup_info.get("insight", "")
    if insight:
        parts.append(f"【首发已锁定】{insight}")

    delta_home = (home_multiplier - 1.0) * 100
    delta_away = (away_multiplier - 1.0) * 100
    if abs(delta_home) > 1 or abs(delta_away) > 1:
        parts.append(f"临场修正(主{delta_home:+.1f}%/客{delta_away:+.1f}%)")

    return "；".join(parts)


def _generate_bet_advice(pred: str, conf: float, env_features: dict) -> str:
    has_warning = len(env_features.get("warnings", [])) > 1 or (
        len(env_features["warnings"]) == 1 and "环境条件中性" not in env_features["warnings"][0]
    )

    if conf >= 60:
        return "高置信度推荐" if not has_warning else "置信度高但需注意环境风险"
    elif conf >= 48:
        return "中等信心可参考" if not has_warning else "中等信心，环境变数较大"
    else:
        return "建议观望" if has_warning else "小额娱乐"


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
    home_as: float,
    away_as: float,
    home_dw: float,
    away_dw: float,
    env_features: dict,
    home_injuries: float = 0,
    away_injuries: float = 0,
) -> str:
    """Generate a varied, style-aware match simulation narrative."""
    warnings = env_features.get("warnings", [])
    rank_gap = abs(home_rank - away_rank)
    tier_gap = abs(_tier(home_rank) - _tier(away_rank))

    # --- Opening phase (style-aware) ---
    # High attack vs weak defence → aggressive opening
    if home_as > 1.2 and away_dw > 1.3:
        opening = f"{home_team}开场即展现强大攻击火力，持续施压{away_team}防线。"
    elif away_as > 1.2 and home_dw > 1.3:
        opening = f"{away_team}反客为主，利用{home_team}防守漏洞频频制造威胁。"
    elif home_as < 0.8 and away_as < 0.8:
        opening = "双方进攻效率均偏低，开局阶段以中场绞杀为主，破门机会寥寥。"
    elif any("高海拔" in w for w in warnings):
        opening = f"在{env_features['elevation_m']}米高原球场，客队明显不适，主队从开场便掌控局面。"
    elif any("湿热" in w for w in warnings):
        opening = f"高温{env_features['temp_c']}°C+高湿{env_features['humidity_pct']}%天气下，双方刻意放慢节奏。"
    elif any("降水" in w for w in warnings):
        opening = "湿滑场地让地面传导变得困难，双方开场以长传试探为主。"
    else:
        opening = "双方开场后迅速进入比赛节奏，中场争夺激烈。"

    # --- Mid-game phase ---
    mid_parts: list[str] = []

    if tier_gap >= 3:
        mid_parts.append(
            f"双方实力档次差距明显（{_tier_label(home_rank)} vs {_tier_label(away_rank)}），"
            f"{home_team if home_rank < away_rank else away_team}的技战术优势全面压制对手。"
        )
    elif tier_gap >= 2:
        mid_parts.append(
            f"存在明显的档次差距，"
            f"{home_team if home_rank < away_rank else away_team}掌控比赛节奏，控球率领先。"
        )
    elif rank_gap >= 15:
        mid_parts.append(f"排名差距{rank_gap}位，强队逐步建立场面优势。")
    else:
        mid_parts.append("双方实力接近，比赛处于胶着状态，关键球的处理将决定走向。")

    if home_injuries >= 2:
        mid_parts.append(f"{home_team}核心球员缺阵影响进攻组织。")
    if away_injuries >= 2:
        mid_parts.append(f"{away_team}伤停严重，防线轮转出现漏洞。")

    if any("长途" in w or "疲劳" in w for w in warnings):
        mid_parts.append("长途飞行带来的疲劳在下半场逐渐显现。")
    if any("草皮" in w for w in warnings):
        mid_parts.append("球场草皮条件对技术型球员的发挥产生微妙影响。")

    mid = "".join(mid_parts)

    # --- Goal scenario (style-driven) ---
    if pred_r == "H":
        if ph >= 0.55 and home_as > 1.2:
            scenario = (
                f"{home_team}凭借压倒性优势早早确立领先，"
                f"比赛变成半场攻防演练，最终以明显优势取胜。"
            )
        elif tier_gap >= 2:
            scenario = (
                f"第{_random.randint(40, 60)}分钟{home_team}打破僵局后控制节奏，"
                f"客队无力反扑，主队稳稳拿下三分。"
            )
        else:
            scenario = (
                f"第{_random.randint(60, 78)}分钟，{home_team}抓住关键机会破门，"
                f"随后众志成城守住胜果。"
            )
    elif pred_r == "A":
        if pa >= 0.55 and away_as > 1.2:
            scenario = (
                f"{away_team}反客为主，凭借高效的进攻转化率，"
                f"在客场带走一场令人信服的胜利。"
            )
        elif tier_gap >= 2:
            scenario = (
                f"{away_team}展现强者风范，早早取得领先后稳扎稳打，"
                f"主队虽有主场之利但难以撼动对手防线。"
            )
        else:
            scenario = (
                f"第{_random.randint(55, 72)}分钟，{away_team}利用反击机会一击致命，"
                f"客场全身而退。"
            )
    else:
        if home_as + away_as > 2.5:
            scenario = "双方大打对攻，各入一球后均有机会超出比分但未能把握，最终平分秋色。"
        elif home_as + away_as < 1.5:
            scenario = "双方进攻乏力，整场比赛破门机会屈指可数，0-0收场。"
        else:
            scenario = "双方各进一球后陷入中场拉锯，最终握手言和。"

    return f"{opening}{mid}{scenario}"


def _tier_label(rank: int) -> str:
    t = _tier(rank)
    return {1: "顶级", 2: "一流", 3: "二流", 4: "发展中"}.get(t, "未知")


# ---------------------------------------------------------------------------
# Main prediction entry-point
# ---------------------------------------------------------------------------

def predict_match(
    home_team: str,
    away_team: str,
    match_date: str,
    stadium_id: str,
    weather_override: Optional[dict] = None,
    injury_override: Optional[dict[str, Optional[dict]]] = None,
) -> dict:
    """Predict a single match with full Poisson + sanity-check pipeline."""

    # --- Resolve base data ----------------------------------------------
    home_rank = _get_fifa_rank(home_team) or 25
    away_rank = _get_fifa_rank(away_team) or 25
    home_sentiment = _get_sentiment(home_team)
    away_sentiment = _get_sentiment(away_team)
    home_heat = _get_social_heat(home_team)
    away_heat = _get_social_heat(away_team)

    # Attacking / defensive coefficients
    home_as = _get_attacking_strength(home_team)
    away_as = _get_attacking_strength(away_team)
    home_dw = _get_defensive_weakness(home_team)
    away_dw = _get_defensive_weakness(away_team)

    # Environmental features
    env_features = get_env_features(stadium_id, home_team, away_team, weather_override)

    # Injury / form
    home_inj = injury_override.get(home_team) if injury_override else None
    away_inj = injury_override.get(away_team) if injury_override else None
    home_form = home_inj.get("form_index", 0.5) if home_inj else 0.5
    away_form = away_inj.get("form_index", 0.5) if away_inj else 0.5
    home_core_inj = home_inj.get("core_injuries", 0) if home_inj else 0
    away_core_inj = away_inj.get("core_injuries", 0) if away_inj else 0
    home_team_fatigue = home_inj.get("fatigue_index", 0.3) if home_inj else 0.3
    away_team_fatigue = away_inj.get("fatigue_index", 0.3) if away_inj else 0.3

    # --- Part A: H/D/A via weighted ensemble ---------------------------
    ph, pd, pa = _compute_base_probabilities(
        home_rank, away_rank,
        home_sentiment, away_sentiment,
        home_heat, away_heat,
        env_features,
        home_form=home_form, away_form=away_form,
        home_injuries=home_core_inj, away_injuries=away_core_inj,
        home_fatigue=home_team_fatigue, away_fatigue=away_team_fatigue,
    )

    # --- Part B: Sanity check (熔断机制) --------------------------------
    ph, pd, pa = _apply_sanity_check(
        ph, pd, pa, home_rank, away_rank, home_team, away_team,
    )

    # --- Part C: Determine result --------------------------------------
    if ph >= pd and ph >= pa:
        pred, pred_r, conf = "主胜", "H", ph
    elif pa >= ph and pa >= pd:
        pred, pred_r, conf = "客胜", "A", pa
    else:
        pred, pred_r, conf = "平局", "D", pd

    # --- Part D: Poisson score probabilities ---------------------------
    # Expected goals from AS / DW
    lambda_h = _expected_goals(home_as, away_dw, is_home=True)
    lambda_a = _expected_goals(away_as, home_dw, is_home=False)

    # --- Part D2: Live-Lineup multipliers (v4.0.0) ---------------------
    match_key = f"{match_date}_{home_team}_{away_team}"
    lineup_info = get_lineup_multipliers(home_team, away_team, match_key)
    prediction_status = lineup_info["prediction_status"]

    if prediction_status == "Live-Lineup":
        lambda_h *= lineup_info["home_multiplier"]
        lambda_a *= lineup_info["away_multiplier"]

    # Tilt lambdas toward the ensemble result
    if pred_r == "H":
        lambda_h *= (0.85 + ph * 0.35)
        lambda_a *= (1.05 - ph * 0.15)
    elif pred_r == "A":
        lambda_a *= (0.85 + pa * 0.35)
        lambda_h *= (1.05 - pa * 0.15)
    else:
        avg = (lambda_h + lambda_a) / 2
        mix = 0.3 + pd * 0.6
        lambda_h = lambda_h * (1 - mix) + avg * mix
        lambda_a = lambda_a * (1 - mix) + avg * mix

    # Recompute H/D/A with lineup-adjusted lambdas when Live-Lineup
    if prediction_status == "Live-Lineup":
        total_goals = lambda_h + lambda_a
        adj_home = 0.38 + (lambda_h - lambda_a) / max(total_goals, 0.5) * 0.25
        adj_away = 0.27 - (lambda_h - lambda_a) / max(total_goals, 0.5) * 0.25
        adj_draw = 1.0 - adj_home - adj_away
        adj_home = max(0.03, min(0.70, adj_home))
        adj_away = max(0.03, min(0.70, adj_away))
        adj_total = adj_home + adj_draw + adj_away
        l_ph, l_pd, l_pa = adj_home / adj_total, adj_draw / adj_total, adj_away / adj_total

        # Blend live-lineup HDA with ensemble HDA (60% live / 40% ensemble)
        ph = ph * 0.4 + l_ph * 0.6
        pd = pd * 0.4 + l_pd * 0.6
        pa = pa * 0.4 + l_pa * 0.6

        # Re-determine result
        if ph >= pd and ph >= pa:
            pred, pred_r, conf = "主胜", "H", ph
        elif pa >= ph and pa >= pd:
            pred, pred_r, conf = "客胜", "A", pa
        else:
            pred, pred_r, conf = "平局", "D", pd

    score_probs = _bivariate_poisson_matrix(lambda_h, lambda_a)

    # --- Part E: Narrative simulation ----------------------------------
    simulation = _generate_match_simulation(
        home_team, away_team, pred, pred_r, ph, pd, pa,
        home_rank, away_rank,
        home_as, away_as, home_dw, away_dw,
        env_features, home_injuries=home_core_inj, away_injuries=away_core_inj,
    )

    # --- Part F: Reason & advice ---------------------------------------
    reason = _generate_reason_v4(
        pred, conf * 100, home_team, away_team, home_rank, away_rank,
        env_features, home_core_inj, away_core_inj,
        prediction_status=prediction_status,
        lineup_info=lineup_info.get("lineup_details"),
        home_multiplier=lineup_info.get("home_multiplier", 1.0),
        away_multiplier=lineup_info.get("away_multiplier", 1.0),
    )
    bet_advice = _generate_bet_advice(pred, conf * 100, env_features)

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
        "home_as": round(home_as, 2),
        "away_as": round(away_as, 2),
        "home_dw": round(home_dw, 2),
        "away_dw": round(away_dw, 2),
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
        # v4.0.0 Live-Lineup fields
        "prediction_status": prediction_status,
        "lineup_info": lineup_info.get("lineup_details"),
        "home_formation": lineup_info.get("home_formation"),
        "away_formation": lineup_info.get("away_formation"),
        "home_formation_label": lineup_info.get("home_formation_label"),
        "away_formation_label": lineup_info.get("away_formation_label"),
    }
