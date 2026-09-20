from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "observation_id",
    "neighborhood",
    "observation_date",
    "latitude",
    "longitude",
    "plant_species_count",
    "high_pollen_species_count",
    "pollen_season_factor",
    "bloom_intensity_score",
    "vegetation_cover_pct",
    "temperature_c",
    "humidity_pct",
    "wind_speed_kmh",
    "rainfall_mm_24h",
    "air_quality_index",
    "particulate_matter_ug_m3",
    "historical_pollen_level",
    "weather_stability_score",
    "monitoring_completeness_pct",
]

NUMERIC_COLUMNS = [c for c in REQUIRED_COLUMNS if c not in {"observation_id", "neighborhood", "observation_date"}]


def clip01(value):
    """Clip scalar or vector-like values to [0, 1] without coercing Series to float."""
    return np.clip(value, 0.0, 1.0)


def scale(value, low: float, high: float):
    """Min-max scale scalar or pandas/numpy vector-like inputs into [0, 1]."""
    if high == low:
        return np.zeros_like(value, dtype=float)
    return clip01((value - low) / (high - low))


def classify(score: float) -> str:
    if score >= 75:
        return "Critical"
    if score >= 55:
        return "High"
    if score >= 35:
        return "Moderate"
    return "Low"


def validate_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = validate_columns(out)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    out["observation_date"] = pd.to_datetime(out["observation_date"], errors="coerce")
    if out["observation_date"].isna().any():
        raise ValueError("observation_date contains invalid or missing dates.")

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Keep scores in safe analytical ranges.
    out["plant_species_count"] = out["plant_species_count"].clip(lower=0)
    out["high_pollen_species_count"] = out["high_pollen_species_count"].clip(lower=0)
    bounded = {
        "pollen_season_factor": (0, 1),
        "bloom_intensity_score": (0, 100),
        "vegetation_cover_pct": (0, 100),
        "temperature_c": (-30, 55),
        "humidity_pct": (0, 100),
        "wind_speed_kmh": (0, 150),
        "rainfall_mm_24h": (0, 500),
        "air_quality_index": (0, 500),
        "particulate_matter_ug_m3": (0, 500),
        "historical_pollen_level": (0, 100),
        "weather_stability_score": (0, 100),
        "monitoring_completeness_pct": (0, 100),
    }
    for col, (lo, hi) in bounded.items():
        out[col] = out[col].clip(lower=lo, upper=hi)

    if out[NUMERIC_COLUMNS].isna().any().any():
        bad = [c for c in NUMERIC_COLUMNS if out[c].isna().any()]
        raise ValueError("Numeric fields contain missing/non-numeric values: " + ", ".join(bad))

    out = add_derived_metrics(out)
    return out


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    species_pressure = 0.55 * scale(out["high_pollen_species_count"], 0, 8) + 0.45 * scale(out["plant_species_count"], 0, 20)
    bloom_pressure = 0.65 * scale(out["bloom_intensity_score"], 0, 100) + 0.35 * out["pollen_season_factor"]
    weather_pressure = (
        0.45 * scale(out["wind_speed_kmh"], 2, 28)
        + 0.25 * scale(out["temperature_c"], 12, 32)
        + 0.30 * scale(out["weather_stability_score"], 0, 100)
    )
    pollution_pressure = (
        0.55 * scale(out["air_quality_index"], 0, 200)
        + 0.45 * scale(out["particulate_matter_ug_m3"], 5, 80)
    )
    dryness_pressure = 1.0 - scale(out["rainfall_mm_24h"], 0, 25)
    history_pressure = scale(out["historical_pollen_level"], 0, 100)
    vegetation_pressure = scale(out["vegetation_cover_pct"], 0, 80)

    # Rain can temporarily suppress airborne pollen, while light winds may keep pollen suspended.
    out["species_pressure_score"] = (species_pressure * 100).round(2)
    out["bloom_pressure_score"] = (bloom_pressure * 100).round(2)
    out["weather_pressure_score"] = (0.60 * weather_pressure + 0.40 * dryness_pressure) * 100
    out["pollution_pressure_score"] = pollution_pressure * 100
    out["vegetation_pressure_score"] = vegetation_pressure * 100
    out["historical_pressure_score"] = history_pressure * 100

    raw = (
        0.22 * species_pressure
        + 0.18 * bloom_pressure
        + 0.18 * weather_pressure
        + 0.15 * pollution_pressure
        + 0.10 * vegetation_pressure
        + 0.12 * history_pressure
        + 0.05 * (1.0 - out["monitoring_completeness_pct"] / 100.0)
    ) * 100

    # A rain event reduces airborne pressure; cap the reduction to keep the score explainable.
    rain_reduction = np.minimum(out["rainfall_mm_24h"] / 35.0, 0.22) * 100
    out["rain_suppression_score"] = rain_reduction.round(2)
    out["pollen_risk_score"] = np.clip(raw - rain_reduction * 0.22, 0, 100).round(1)
    out["risk_class"] = out["pollen_risk_score"].apply(classify)

    component_cols = [
        "species_pressure_score",
        "bloom_pressure_score",
        "weather_pressure_score",
        "pollution_pressure_score",
        "vegetation_pressure_score",
        "historical_pressure_score",
    ]
    out["dominant_driver"] = out[component_cols].idxmax(axis=1).str.replace("_score", "", regex=False).str.replace("_", " ").str.title()
    out["review_flag"] = np.where(
        (out["pollen_risk_score"] >= 55)
        | (out["monitoring_completeness_pct"] < 70),
        "Review",
        "Monitor",
    )
    return out


def summarize(df: pd.DataFrame) -> dict[str, float | int]:
    return {
        "observations": int(len(df)),
        "neighborhoods": int(df["neighborhood"].nunique()),
        "average_risk": float(df["pollen_risk_score"].mean()),
        "high_or_critical": int(df["risk_class"].isin(["High", "Critical"]).sum()),
        "review_count": int((df["review_flag"] == "Review").sum()),
        "avg_aqi": float(df["air_quality_index"].mean()),
    }


def scenario_score(
    row: pd.Series,
    season_delta: float = 0.0,
    bloom_delta: float = 0.0,
    wind_delta: float = 0.0,
    pollution_delta: float = 0.0,
    rainfall_delta: float = 0.0,
) -> float:
    payload = row.copy()
    payload["pollen_season_factor"] = np.clip(float(payload["pollen_season_factor"]) + season_delta, 0, 1)
    payload["bloom_intensity_score"] = np.clip(float(payload["bloom_intensity_score"]) + bloom_delta, 0, 100)
    payload["wind_speed_kmh"] = max(0.0, float(payload["wind_speed_kmh"]) + wind_delta)
    payload["air_quality_index"] = np.clip(float(payload["air_quality_index"]) + pollution_delta, 0, 500)
    payload["rainfall_mm_24h"] = max(0.0, float(payload["rainfall_mm_24h"]) + rainfall_delta)
    frame = pd.DataFrame([payload[REQUIRED_COLUMNS]])
    return float(prepare_data(frame)["pollen_risk_score"].iloc[0])
