from pathlib import Path

import pandas as pd

from analytics import REQUIRED_COLUMNS, classify, prepare_data, scenario_score


def load_sample():
    path = Path(__file__).resolve().parents[1] / "data" / "sample_pollen_allergy.csv"
    return pd.read_csv(path)


def test_sample_has_required_columns_and_scores():
    df = prepare_data(load_sample())
    assert not [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert df["pollen_risk_score"].between(0, 100).all()
    assert df["risk_class"].isin(["Low", "Moderate", "High", "Critical"]).all()


def test_classification_boundaries():
    assert classify(0) == "Low"
    assert classify(35) == "Moderate"
    assert classify(55) == "High"
    assert classify(75) == "Critical"


def test_scenario_is_deterministic_and_sensitive():
    df = prepare_data(load_sample())
    row = df.iloc[0]
    baseline = scenario_score(row)
    hotter_and_bloomier = scenario_score(row, season_delta=0.25, bloom_delta=25, pollution_delta=30)
    assert baseline == scenario_score(row)
    assert hotter_and_bloomier >= baseline


def test_missing_column_error():
    raw = load_sample().drop(columns=["air_quality_index"])
    try:
        prepare_data(raw)
    except ValueError as exc:
        assert "air_quality_index" in str(exc)
    else:
        raise AssertionError("Expected missing-column validation error")
