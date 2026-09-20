from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import REQUIRED_COLUMNS, add_derived_metrics, prepare_data, scenario_score, summarize

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
ASSET_DIR = BASE / "assets"

st.set_page_config(
    page_title="Pollen Allergy Risk Forecaster",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#203040; --muted:#627283; --line:#dce5ee; --card:#ffffff; --soft:#f5f8fb; }
    .stApp { background: linear-gradient(180deg,#f7fbff 0%,#f2f7fb 52%,#eef5f1 100%); color:var(--ink); }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1500px; }
    [data-testid="stSidebar"] { background: #ffffff; border-right:1px solid var(--line); }
    .hero { background: linear-gradient(135deg,#ecfff7 0%,#eef7ff 55%,#f9fbff 100%); border:1px solid #d7e7e2; border-radius:24px; padding:22px 26px; box-shadow:0 14px 30px rgba(42,72,104,.07); }
    .hero-title { font-size:2.15rem; font-weight:800; letter-spacing:-.03em; margin:0; color:#183246; }
    .hero-sub { margin:.45rem 0 0; color:#526575; font-size:1.02rem; line-height:1.55; }
    .eyebrow { font-size:.76rem; font-weight:800; text-transform:uppercase; letter-spacing:.14em; color:#4d806f; margin-bottom:.55rem; }
    .card { background:var(--card); border:1px solid var(--line); border-radius:18px; padding:16px 18px; box-shadow:0 10px 22px rgba(43,66,90,.05); }
    .kpi-label { color:#687989; font-size:.82rem; font-weight:700; }
    .kpi-value { color:#173246; font-size:1.7rem; font-weight:800; margin-top:.15rem; }
    .kpi-note { color:#778899; font-size:.76rem; }
    .section-title { color:#193447; font-size:1.25rem; font-weight:800; margin:.2rem 0 .6rem; }
    .section-sub { color:#667685; margin-bottom: .8rem; }
    .pill { display:inline-block; padding:.26rem .62rem; border-radius:999px; font-size:.75rem; font-weight:800; margin-right:.35rem; border:1px solid #d9e3ea; background:#f7fafc; }
    .warning { background:#fff8e7; border:1px solid #efdca4; padding:12px 14px; border-radius:14px; color:#6c5314; }
    .success { background:#edf9f4; border:1px solid #c7e9d7; padding:12px 14px; border-radius:14px; color:#246044; }
    .footer-note { color:#718191; font-size:.78rem; line-height:1.5; }
    </style>
    """,
    unsafe_allow_html=True,
)


def local_svg(name: str) -> str:
    path = ASSET_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def card_metric(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def md_table(frame: pd.DataFrame, max_rows: int = 12) -> str:
    view = frame.head(max_rows).copy()
    if view.empty:
        return "_No rows available._"
    cols = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for value in row.tolist():
            text = str(value).replace("|", "\\|").replace("\n", " ")
            vals.append(text)
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return prepare_data(pd.read_csv(DATA_DIR / "sample_pollen_allergy.csv"))


with st.sidebar:
    st.markdown('<div class="eyebrow">LOCAL-FIRST • AIR & ENVIRONMENT</div>', unsafe_allow_html=True)
    st.header("Workspace")
    upload = st.file_uploader("Upload neighborhood CSV", type=["csv"], help="CSV is processed locally in this app.")
    if upload is not None:
        try:
            base_df = prepare_data(pd.read_csv(upload))
            st.success(f"Loaded {len(base_df):,} observations")
        except Exception as exc:
            st.error(str(exc))
            st.stop()
    else:
        base_df = load_sample()
        st.info("Using bundled synthetic sample data.")

    st.divider()
    page = st.radio(
        "Navigate",
        ["Overview", "Neighborhood Map", "Pollen Drivers", "Weather & Pollution", "Seasonality", "Priority Review", "Scenario Lab", "Reports & Export", "Data Explorer"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("No external APIs required. All calculations run in Python on local data.")

summary = summarize(base_df)

st.markdown(
    '<div class="hero"><div class="eyebrow">🌿 NEIGHBORHOOD ALLERGY INTELLIGENCE</div><div class="hero-title">Pollen Allergy Risk Forecaster</div><div class="hero-sub">A local-first analytical workspace for screening pollen-risk pressure from plant, weather, wind, seasonality and air-pollution signals — designed for transparent environmental planning, not medical diagnosis.</div></div>',
    unsafe_allow_html=True,
)
st.write("")

if page == "Overview":
    cols = st.columns(6)
    values = [
        ("Observations", f"{summary['observations']:,}", "local records"),
        ("Neighborhoods", f"{summary['neighborhoods']:,}", "coverage areas"),
        ("Average risk", f"{summary['average_risk']:.1f}/100", "screening pressure"),
        ("High / Critical", f"{summary['high_or_critical']:,}", "records"),
        ("Review flags", f"{summary['review_count']:,}", "priority follow-up"),
        ("Avg AQI", f"{summary['avg_aqi']:.0f}", "pollution signal"),
    ]
    for col, item in zip(cols, values):
        with col:
            card_metric(*item)

    st.write("")
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="section-title">Risk trend across observations</div>', unsafe_allow_html=True)
        daily = base_df.sort_values("observation_date").groupby("observation_date", as_index=False)["pollen_risk_score"].mean()
        fig = px.area(daily, x="observation_date", y="pollen_risk_score", markers=True, labels={"pollen_risk_score":"Average pollen-risk score", "observation_date":"Date"})
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=25,b=10), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Risk-class distribution</div>', unsafe_allow_html=True)
        dist = base_df["risk_class"].value_counts().reindex(["Low", "Moderate", "High", "Critical"], fill_value=0).reset_index()
        dist.columns = ["risk_class", "count"]
        fig = px.bar(dist, x="risk_class", y="count", text="count", labels={"risk_class":"Class", "count":"Records"})
        fig.update_traces(textposition="outside")
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=25,b=10), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">How the score is interpreted</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><span class="pill">Low</span><span class="pill">Moderate</span><span class="pill">High</span><span class="pill">Critical</span><p class="footer-note">The 0–100 score is an explainable screening signal combining plant-species pressure, bloom intensity, weather, pollution, vegetation, historical pollen and monitoring completeness. It is not a clinical forecast of symptoms or a medical risk diagnosis.</p></div>', unsafe_allow_html=True)

elif page == "Neighborhood Map":
    st.markdown('<div class="section-title">Neighborhood risk map</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Local coordinate visualization with hoverable risk and environmental attributes. No map tiles or external geocoding are used.</div>', unsafe_allow_html=True)
    palette = {"Low":"#4D9A78", "Moderate":"#E0A63A", "High":"#E77C4A", "Critical":"#C8585A"}
    fig = px.scatter(
        base_df,
        x="longitude",
        y="latitude",
        color="risk_class",
        size="pollen_risk_score",
        hover_name="neighborhood",
        hover_data=["pollen_risk_score","dominant_driver","air_quality_index","wind_speed_kmh"],
        labels={"longitude":"Longitude", "latitude":"Latitude"},
        height=520,
        color_discrete_map=palette,
    )
    fig.update_traces(marker_line_width=1.2, marker_line_color="white")
    fig.update_layout(
        margin=dict(l=10,r=10,t=25,b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#e7eef4"),
        yaxis=dict(showgrid=True, gridcolor="#e7eef4", scaleanchor="x", scaleratio=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Neighborhood comparison</div>', unsafe_allow_html=True)
    nb = base_df.groupby("neighborhood", as_index=False).agg(
        average_risk=("pollen_risk_score","mean"),
        max_risk=("pollen_risk_score","max"),
        avg_aqi=("air_quality_index","mean"),
        avg_bloom=("bloom_intensity_score","mean"),
        avg_wind=("wind_speed_kmh","mean"),
        observations=("observation_id","count"),
    ).sort_values("average_risk", ascending=False)
    st.dataframe(nb.style.format({"average_risk":"{:.1f}","max_risk":"{:.1f}","avg_aqi":"{:.0f}","avg_bloom":"{:.1f}","avg_wind":"{:.1f}"}), use_container_width=True, hide_index=True)

elif page == "Pollen Drivers":
    st.markdown('<div class="section-title">What is driving pollen pressure?</div>', unsafe_allow_html=True)
    comp = pd.DataFrame({
        "driver":["Plant species","Bloom intensity","Weather","Pollution","Vegetation","Historical pollen"],
        "mean_score":[base_df[c].mean() for c in ["species_pressure_score","bloom_pressure_score","weather_pressure_score","pollution_pressure_score","vegetation_pressure_score","historical_pressure_score"]],
    }).sort_values("mean_score", ascending=False)
    fig = px.bar(comp, x="mean_score", y="driver", orientation="h", text="mean_score", labels={"mean_score":"Average driver pressure", "driver":"Driver"})
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(height=380, margin=dict(l=10,r=80,t=25,b=10), plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    l, r = st.columns(2)
    with l:
        st.markdown('<div class="section-title">High-pollen species vs risk</div>', unsafe_allow_html=True)
        fig = px.scatter(base_df, x="high_pollen_species_count", y="pollen_risk_score", size="bloom_intensity_score", color="risk_class", hover_name="neighborhood", labels={"high_pollen_species_count":"High-pollen species count", "pollen_risk_score":"Risk score"})
        fig.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with r:
        st.markdown('<div class="section-title">Dominant drivers</div>', unsafe_allow_html=True)
        driver = base_df["dominant_driver"].value_counts().reset_index()
        driver.columns=["driver","count"]
        fig = px.pie(driver, names="driver", values="count", hole=.55)
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=20,b=10), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Weather & Pollution":
    st.markdown('<div class="section-title">Weather and air-quality pressure</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        fig = px.scatter(base_df, x="wind_speed_kmh", y="pollen_risk_score", color="air_quality_index", size="vegetation_cover_pct", hover_name="neighborhood", labels={"wind_speed_kmh":"Wind speed (km/h)", "pollen_risk_score":"Risk score", "air_quality_index":"AQI"})
        fig.update_layout(height=390, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.scatter(base_df, x="air_quality_index", y="pollen_risk_score", color="humidity_pct", size="particulate_matter_ug_m3", hover_name="neighborhood", labels={"air_quality_index":"Air-quality index", "pollen_risk_score":"Risk score", "humidity_pct":"Humidity (%)"})
        fig.update_layout(height=390, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    weather = base_df.groupby(pd.cut(base_df["wind_speed_kmh"], bins=[-1,5,12,20,35,200], labels=["0–5","5–12","12–20","20–35","35+"])).agg(avg_risk=("pollen_risk_score","mean"), count=("observation_id","count")).reset_index()
    st.markdown('<div class="section-title">Risk by wind-speed band</div>', unsafe_allow_html=True)
    st.dataframe(weather.style.format({"avg_risk":"{:.1f}"}), use_container_width=True, hide_index=True)

elif page == "Seasonality":
    st.markdown('<div class="section-title">Seasonality and bloom pattern</div>', unsafe_allow_html=True)
    month = base_df.assign(month=base_df["observation_date"].dt.strftime("%b"), month_num=base_df["observation_date"].dt.month).groupby(["month_num","month"], as_index=False).agg(avg_risk=("pollen_risk_score","mean"), avg_bloom=("bloom_intensity_score","mean"), avg_history=("historical_pollen_level","mean"), observations=("observation_id","count")).sort_values("month_num")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=month["month"], y=month["avg_risk"], mode="lines+markers", name="Risk"))
    fig.add_trace(go.Scatter(x=month["month"], y=month["avg_bloom"], mode="lines+markers", name="Bloom"))
    fig.add_trace(go.Scatter(x=month["month"], y=month["avg_history"], mode="lines+markers", name="Historical pollen"))
    fig.update_layout(height=400, margin=dict(l=10,r=10,t=25,b=10), plot_bgcolor="white", paper_bgcolor="white", yaxis_title="Average score")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(month.style.format({"avg_risk":"{:.1f}","avg_bloom":"{:.1f}","avg_history":"{:.1f}"}), use_container_width=True, hide_index=True)

elif page == "Priority Review":
    st.markdown('<div class="section-title">Priority review queue</div>', unsafe_allow_html=True)
    threshold = st.slider("Review threshold", 35, 90, 55)
    priority = base_df[(base_df["pollen_risk_score"] >= threshold) | (base_df["monitoring_completeness_pct"] < 70)].copy()
    priority["priority_reason"] = np.where(priority["monitoring_completeness_pct"] < 70, "Low monitoring completeness", priority["dominant_driver"] + " pressure")
    priority = priority.sort_values(["pollen_risk_score","monitoring_completeness_pct"], ascending=[False, True])
    st.metric("Records needing review", len(priority))
    st.dataframe(priority[["observation_id","neighborhood","observation_date","pollen_risk_score","risk_class","dominant_driver","monitoring_completeness_pct","priority_reason"]], use_container_width=True, hide_index=True)

elif page == "Scenario Lab":
    st.markdown('<div class="section-title">Scenario Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Explore transparent what-if changes to seasonality, bloom, wind, pollution and recent rainfall for one local observation.</div>', unsafe_allow_html=True)
    chosen = st.selectbox("Observation", base_df["observation_id"].tolist())
    row = base_df.loc[base_df["observation_id"] == chosen].iloc[0]
    c1,c2,c3 = st.columns(3)
    with c1:
        season_delta = st.slider("Pollen season factor change", -0.30, 0.30, 0.00, 0.05)
        bloom_delta = st.slider("Bloom intensity change", -40.0, 40.0, 0.0, 5.0)
    with c2:
        wind_delta = st.slider("Wind-speed change (km/h)", -15.0, 20.0, 0.0, 1.0)
        pollution_delta = st.slider("AQI change", -80.0, 120.0, 0.0, 5.0)
    with c3:
        rainfall_delta = st.slider("24h rainfall change (mm)", 0.0, 60.0, 0.0, 5.0)
        show_detail = st.checkbox("Show calculation details", value=True)
    before = float(row["pollen_risk_score"])
    after = scenario_score(row, season_delta, bloom_delta, wind_delta, pollution_delta, rainfall_delta)
    d1,d2,d3 = st.columns(3)
    with d1: card_metric("Baseline", f"{before:.1f}/100", row["risk_class"])
    with d2: card_metric("Scenario", f"{after:.1f}/100", "modeled result")
    with d3: card_metric("Change", f"{after-before:+.1f}", "score points")
    if show_detail:
        st.markdown('<div class="card"><b>Scenario basis</b><br><span class="footer-note">This is a deterministic model on the selected row using the same score function as the dashboard. It is not a medical prediction and does not ingest live weather or pollen feeds.</span></div>', unsafe_allow_html=True)

elif page == "Reports & Export":
    st.markdown('<div class="section-title">Reports & export</div>', unsafe_allow_html=True)
    report = base_df.sort_values("pollen_risk_score", ascending=False)[["observation_id","neighborhood","observation_date","pollen_risk_score","risk_class","dominant_driver","air_quality_index","wind_speed_kmh"]]
    summary_text = (
        "# Pollen Allergy Risk Forecaster — Local Screening Report\n\n"
        f"Observations: {len(base_df)}  \nNeighborhoods: {base_df['neighborhood'].nunique()}  \n"
        f"Average screening score: {base_df['pollen_risk_score'].mean():.1f}/100  \n"
        f"High/Critical observations: {int(base_df['risk_class'].isin(['High','Critical']).sum())}\n\n"
        "## Priority records\n\n"
        + md_table(report, 15)
        + "\n\n## Responsible-use note\n\n"
        "This local screening tool identifies environmental combinations that may warrant additional review. It does not diagnose pollen allergy, predict individual symptoms, or replace medical advice."
    )
    st.download_button("Download Markdown report", summary_text, file_name="pollen_allergy_local_report.md", mime="text/markdown")
    csv_bytes = report.to_csv(index=False).encode("utf-8")
    st.download_button("Download priority CSV", csv_bytes, file_name="pollen_priority_review.csv", mime="text/csv")
    st.code(summary_text[:5000], language="markdown")

elif page == "Data Explorer":
    st.markdown('<div class="section-title">Local data explorer</div>', unsafe_allow_html=True)
    neighborhoods = st.multiselect("Neighborhood filter", sorted(base_df["neighborhood"].unique().tolist()))
    classes = st.multiselect("Risk class filter", ["Low","Moderate","High","Critical"], default=["Low","Moderate","High","Critical"])
    view = base_df.copy()
    if neighborhoods:
        view = view[view["neighborhood"].isin(neighborhoods)]
    view = view[view["risk_class"].isin(classes)]
    st.caption(f"Showing {len(view):,} of {len(base_df):,} records")
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.markdown('<div class="section-title">Data quality</div>', unsafe_allow_html=True)
    quality = pd.DataFrame({
        "field": REQUIRED_COLUMNS,
        "missing": [int(view[c].isna().sum()) for c in REQUIRED_COLUMNS],
        "unique": [int(view[c].nunique()) for c in REQUIRED_COLUMNS],
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)

st.markdown("<div class='footer-note'>Local-first design • bundled synthetic sample data • analytical screening only • no external APIs or live pollen feeds required</div>", unsafe_allow_html=True)
