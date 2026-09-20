# Pollen Allergy Risk Forecaster

A privacy-conscious, local-first Streamlit analytics dashboard that screens neighborhood-level pollen-risk pressure using plant species, bloom intensity, weather, wind, seasonality, rainfall, and air-pollution signals.

## Features
- Explainable 0–100 screening score and Low / Moderate / High / Critical classification
- Neighborhood local-coordinate visualization without external map tiles
- Plant-species, bloom, weather, wind, rainfall, AQI and PM analytics
- Priority review queue and data-completeness flags
- Deterministic what-if Scenario Lab
- Local Markdown and CSV exports
- Bundled synthetic sample data

## Local run
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest tests/ -q
streamlit run app.py
```

The system does not diagnose allergy, predict individual symptoms, or replace clinical advice. It is an environmental screening and planning tool.
