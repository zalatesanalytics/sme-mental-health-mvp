import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# PAGE CONFIG + SMART BACKGROUND
# --------------------------------------------------------------------
st.set_page_config(
    page_title="LaborPulse-AI Improve SME Productivity",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #eef3ff 0, #ffffff 55%);
}
.main-block {
    background-color: rgba(255, 255, 255, 0.97);
    padding: 1.5rem;
    border-radius: 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
.title-text {
    font-size: 36px !important;
    font-weight: 700 !important;
    padding-top: 0.5rem;
}
.subtitle-text {
    font-size: 18px !important;
    color: #444444 !important;
    margin-top: -10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-block">', unsafe_allow_html=True)

# --------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/Flag_of_Canada.svg/320px-Flag_of_Canada.svg.png",
        use_column_width=True
    )
with col_title:
    st.markdown('<p class="title-text">LaborPulse-AI Improve SME Productivity</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle-text">AI-powered decision support to reduce mental-health-related productivity loss and strengthen Canada’s SMEs.</p>',
        unsafe_allow_html=True
    )

# --------------------------------------------------------------------
# ABOUT AGENT (GLOBAL)
# --------------------------------------------------------------------
with st.expander("About LaborPulse-AI (AI decision-support agent)", expanded=False):
    st.markdown("""
LaborPulse-AI analyzes **aggregated** SME workforce mental-health indicators to estimate
economic loss, simulate interventions, and support **policy-level** and **organizational**
decisions – never individual profiling.

- **Impact visualization:** dashboards for loss, savings, and lost days.
- **Predictive analytics (MVP):** simple projections of future economic impact based on current patterns.
- **Outcome assessment:** highlights both **gains** (savings) and **losses** (when an intervention backfires).
- **Responsible AI:** only aggregated SME/provincial outputs, no worker-level scoring.
""")

# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------
@st.cache_data
def generate_synthetic_data(intensity="mixed", seed=42, n_smes=200):
    np.random.seed(seed)
    provinces = [
        'Ontario','Quebec','British Columbia','Alberta','Manitoba',
        'Saskatchewan','Nova Scotia','New Brunswick',
        'Newfoundland & Labrador','Prince Edward Island'
    ]
    sectors = ['Tech','Retail','Manufacturing','Services']
    rows = []

    for i in range(n_smes):
        prov = np.random.choice(provinces)
        sector = np.random.choice(sectors)
        employees = int(np.random.randint(10,300))
        avg_daily_wage = float(np.round(np.random.uniform(120,450),2))

        if intensity == "low":
            stress = np.random.randint(1,5)
            burnout = np.random.randint(1,4)
            absenteeism = np.random.randint(1,4)
        elif intensity == "medium":
            stress = np.random.randint(3,7)
            burnout = np.random.randint(3,7)
            absenteeism = np.random.randint(2,6)
        elif intensity == "high":
            stress = np.random.randint(6,11)
            burnout = np.random.randint(6,11)
            absenteeism = np.random.randint(5,11)
        else:
            stress = np.random.randint(1,11)
            burnout = np.random.randint(1,11)
            absenteeism = np.random.randint(1,11)

        rows.append({
            "province": prov,
            "sme_id": f"{prov[:3].upper()}_{i+1:04d}",
            "sector": sector,
            "employees": employees,
            "avg_daily_wage": avg_daily_wage,
            "stress_score": stress,
            "burnout_score": burnout,
            "absenteeism_score": absenteeism
        })
    return pd.DataFrame(rows)


@st.cache_data
def generate_timeseries_data(years=5, shock=False, seed=123, n_smes=150):
    """Synthetic monthly time series for SMEs, with optional economic shock (e.g. COVID-style)."""
    np.random.seed(seed)
    provinces = [
        "Ontario","Quebec","British Columbia","Alberta",
        "Manitoba","Saskatchewan","Nova Scotia","New Brunswick",
        "Newfoundland & Labrador","Prince Edward Island"
    ]
    sectors = ["Tech","Retail","Manufacturing","Services"]
    rows = []

    # SHOCK years: 3-year sustained (e.g., 2020–2022)
    shock_years = [2020, 2021, 2022] if shock else []

    base_year = 2020
    sme_list = []
    for i in range(n_smes):
        prov = np.random.choice(provinces)
        sector = np.random.choice(sectors)
        employees = int(np.random.randint(15,350))
        avg_daily_wage = float(np.round(np.random.uniform(130,420),2))
        sme_list.append((prov, sector, employees, avg_daily_wage))

    for sme_id, (prov, sector, employees, avg_daily_wage) in enumerate(sme_list, start=1):
        for year in range(base_year, base_year + years):
            for month in range(1,13):
                # Baseline scores
                stress = np.random.randint(2,8)
                burnout = np.random.randint(2,8)
                absenteeism = np.random.randint(1,6)

                if year in shock_years:
                    shock_factor = 1.3 if sector in ["Retail","Services"] else 1.1
                    stress += int(np.random.randint(3,6) * shock_factor)
                    burnout += int(np.random.randint(3,6) * shock_factor)
                    absenteeism += int(np.random.randint(3,7) * shock_factor)

                if shock and year >= max(shock_years) + 1:
                    recovery_factor = 0.8
                    stress = max(1, int(stress * recovery_factor))
                    burnout = max(1, int(burnout * recovery_factor))
                    absenteeism = max(1, int(absenteeism * recovery_factor))

                rows.append({
                    "province": prov,
                    "sme_id": f"{prov[:3].upper()}_{sme_id:04d}",
                    "sector": sector,
                    "year": year,
                    "month": month,
                    "employees": employees,
                    "avg_daily_wage": avg_daily_wage,
                    "stress_score": min(stress,10),
                    "burnout_score": min(burnout,10),
                    "absenteeism_score": min(absenteeism,10)
                })
    return pd.DataFrame(rows)


def compute_economic_loss(df_in, workdays_per_month, burnout_to_presenteeism_scale):
    df_l = df_in.copy()
    df_l["absenteeism_days_per_emp"] = df_l["absenteeism_score"].clip(1,10)
    df_l["presenteeism_days_equiv_per_emp"] = (
        (df_l["burnout_score"] / 10.0) * workdays_per_month * burnout_to_presenteeism_scale
    )
    df_l["lost_days_per_emp"] = df_l["absenteeism_days_per_emp"] + df_l["presenteeism_days_equiv_per_emp"]
    df_l["loss_per_emp_cad"] = df_l["lost_days_per_emp"] * df_l["avg_daily_wage"]
    df_l["estimated_monthly_loss_cad"] = df_l["loss_per_emp_cad"] * df_l["employees"]
    df_l["estimated_monthly_loss_cad"] = df_l["estimated_monthly_loss_cad"].round(2)
    return df_l


# --------------------------------------------------------------------
# SIDEBAR NAVIGATION + GLOBAL PARAMETERS
# --------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["SME Productivity Dashboard",
     "Time-Series Trends",
     "Forecasting Sandbox",
     "About & Methodology"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Economic model parameters")
WORKDAYS_PER_MONTH = st.sidebar.number_input("Workdays per month", 15, 25, 20)
BURNOUT_TO_PRESENTEEISM_SCALE = st.sidebar.number_input(
    "Burnout → presenteeism scale", 0.0, 1.0, 0.10
)

# --------------------------------------------------------------------
# PAGE 1 – SME PRODUCTIVITY DASHBOARD
# --------------------------------------------------------------------
if page == "SME Productivity Dashboard":
    st.write("## SME Productivity Dashboard")

    scenario_choice = st.sidebar.radio(
        "Dataset scenario",
        [
            "Upload my own CSV",
            "Sample: Low impact",
            "Sample: Medium impact",
            "Sample: High impact",
            "Sample: Mixed impact"
        ]
    )

    uploaded = None
    if scenario_choice == "Upload my own CSV":
        uploaded = st.sidebar.file_uploader(
            "Upload SME CSV",
            type=["csv","xlsx"],
            help="Required: province, sme_id, sector, employees, avg_daily_wage, stress_score, burnout_score, absenteeism_score"
        )
    else:
        st.sidebar.info("Using synthetic sample data. Switch impact level to see different patterns.")

    def load_dashboard_data():
        if scenario_choice == "Upload my own CSV":
            if uploaded is None:
                st.info("Upload a dataset to continue.")
                return None, "Waiting for upload"
            try:
                if uploaded.name.lower().endswith(".csv"):
                    df_local = pd.read_csv(uploaded)
                else:
                    df_local = pd.read_excel(uploaded)
                return df_local, "User upload"
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return None, "Error"
        if scenario_choice == "Sample: Low impact":
            return generate_synthetic_data("low", 10), "Low impact sample"
        if scenario_choice == "Sample: Medium impact":
            return generate_synthetic_data("medium", 20), "Medium impact sample"
        if scenario_choice == "Sample: High impact":
            return generate_synthetic_data("high", 30), "High impact sample"
        return generate_synthetic_data("mixed", 40), "Mixed impact sample"

    df, dataset_label = load_dashboard_data()
    if df is None:
        st.stop()

    st.write(f"**Current dataset:** {dataset_label}")
    st.dataframe(df.head(10))

    required = [
        "province","sme_id","sector","employees","avg_daily_wage",
        "stress_score","burnout_score","absenteeism_score"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    for col in ["employees","avg_daily_wage","stress_score","burnout_score","absenteeism_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_losses = compute_economic_loss(df, WORKDAYS_PER_MONTH, BURNOUT_TO_PRESENTEEISM_SCALE)

    st.write("### SME-level economic loss (sample)")
    st.dataframe(df_losses.head(15))

    # Baseline provincial aggregation
    prov_agg = df_losses.groupby("province").agg(
        total_smes=("sme_id","nunique"),
        total_employees=("employees","sum"),
        avg_stress=("stress_score","mean"),
        avg_burnout=("burnout_score","mean"),
        avg_absenteeism=("absenteeism_score","mean"),
        avg_lost_days=("lost_days_per_emp","mean"),
        estimated_monthly_loss_cad=("estimated_monthly_loss_cad","sum")
    ).reset_index()
    prov_agg["estimated_monthly_loss_cad"] = prov_agg["estimated_monthly_loss_cad"].round(2)

    st.write("### Baseline – Provincial summary")
    st.dataframe(prov_agg.sort_values("estimated_monthly_loss_cad", ascending=False))

    baseline_total = prov_agg["estimated_monthly_loss_cad"].sum()
    total_employees = prov_agg["total_employees"].sum()

    st.metric("National monthly economic loss (baseline)", f"${baseline_total:,.0f}")
    st.metric("Total employees represented", f"{int(total_employees):,}")

    # Bar chart
    st.write("### National loss profile by province")
    fig, ax = plt.subplots(figsize=(8,4))
    ordered = prov_agg.sort_values("estimated_monthly_loss_cad", ascending=True)
    ax.barh(ordered["province"], ordered["estimated_monthly_loss_cad"])
    ax.set_xlabel("Monthly lost productivity (CAD)")
    st.pyplot(fig)

    # Intervention simulation
    st.write("### Intervention simulation")
    intervention = st.selectbox(
        "Intervention type",
        ["None", "Reduce stress scores", "Reduce burnout scores", "Reduce absenteeism scores"]
    )
    reduction_points = st.slider("Reduction points (1–5)", 0, 5, 2)

    def simulate(df_in):
        df_sim = df_in.copy()
        if reduction_points > 0:
            if intervention == "Reduce stress scores":
                df_sim["stress_score"] = (df_sim["stress_score"] - reduction_points).clip(1,10)
            elif intervention == "Reduce burnout scores":
                df_sim["burnout_score"] = (df_sim["burnout_score"] - reduction_points).clip(1,10)
            elif intervention == "Reduce absenteeism scores":
                df_sim["absenteeism_score"] = (df_sim["absenteeism_score"] - reduction_points).clip(1,10)
        df_sim = compute_economic_loss(df_sim, WORKDAYS_PER_MONTH, BURNOUT_TO_PRESENTEEISM_SCALE)
        prov_sim = df_sim.groupby("province").agg(
            total_employees=("employees","sum"),
            avg_lost_days=("lost_days_per_emp","mean"),
            estimated_monthly_loss_cad=("estimated_monthly_loss_cad","sum")
        ).reset_index()
        prov_sim["estimated_monthly_loss_cad"] = prov_sim["estimated_monthly_loss_cad"].round(2)
        return df_sim, prov_sim

    df_sim, prov_sim = simulate(df)

    intervention_total = prov_sim["estimated_monthly_loss_cad"].sum()
    savings = baseline_total - intervention_total
    savings_pct = (savings / baseline_total * 100) if baseline_total > 0 else 0
    label = "Monthly economic gain" if savings >= 0 else "Monthly economic loss"

    st.metric(label, f"${savings:,.0f}", delta=f"{savings_pct:.1f}%")

    st.write("### Provincial impact of intervention")
    impact = prov_agg.merge(prov_sim, on="province", suffixes=("_baseline","_after"))
    impact["cost_reduction_cad"] = (
        impact["estimated_monthly_loss_cad_baseline"] -
        impact["estimated_monthly_loss_cad_after"]
    )
    impact["cost_reduction_pct"] = np.where(
        impact["estimated_monthly_loss_cad_baseline"] > 0,
        impact["cost_reduction_cad"] / impact["estimated_monthly_loss_cad_baseline"] * 100,
        0
    )
    impact["efficiency_gain_cad_per_emp"] = np.where(
        impact["total_employees_baseline"] > 0,
        impact["cost_reduction_cad"] / impact["total_employees_baseline"],
        0
    )
    impact["reduction_lost_days_per_emp"] = (
        impact["avg_lost_days_baseline"] - impact["avg_lost_days_after"]
    )

    impact_display = impact[[
        "province",
        "estimated_monthly_loss_cad_baseline",
        "estimated_monthly_loss_cad_after",
        "cost_reduction_cad",
        "cost_reduction_pct",
        "reduction_lost_days_per_emp",
        "efficiency_gain_cad_per_emp"
    ]]

    st.dataframe(
        impact_display.style.format({
            "estimated_monthly_loss_cad_baseline": "{:,.0f}",
            "estimated_monthly_loss_cad_after": "{:,.0f}",
            "cost_reduction_cad": "{:,.0f}",
            "cost_reduction_pct": "{:,.1f}",
            "reduction_lost_days_per_emp": "{:,.2f}",
            "efficiency_gain_cad_per_emp": "{:,.2f}"
        })
    )

    # Data-driven narrative
    top_row = impact_display.sort_values("cost_reduction_cad", ascending=False).iloc[0]
    nat_eff_gain = impact["efficiency_gain_cad_per_emp"].mean()
    nat_lost_red = impact["reduction_lost_days_per_emp"].mean()

    st.markdown(f"""
**Interpretation (current scenario)**  
- Baseline national loss: **${baseline_total:,.0f} / month**  
- After intervention: **${intervention_total:,.0f} / month**  
- Net change: **${savings:,.0f}** (**{savings_pct:.1f}%**)  
- Most-impacted province: **{top_row['province']}**, with **${top_row['cost_reduction_cad']:,.0f}** change  
- Average efficiency gain: **${nat_eff_gain:,.2f} per employee/month**  
- Average reduction in lost days: **{nat_lost_red:.2f} days/employee/month**

When cost reduction is positive, the intervention generates **economic gains and improved SME productivity**.  
When negative, it signals **increased costs**, helping policymakers redesign or retarget support.
""")

    st.write("### Export results")
    st.download_button(
        "Download provincial impact (CSV)",
        impact_display.to_csv(index=False).encode("utf-8"),
        file_name="laborpulse_prov_results.csv"
    )
    st.download_button(
        "Download SME-level results (CSV)",
        df_sim.to_csv(index=False).encode("utf-8"),
        file_name="laborpulse_sme_results.csv"
    )

# --------------------------------------------------------------------
# PAGE 2 – TIME-SERIES TRENDS
# --------------------------------------------------------------------
elif page == "Time-Series Trends":
    st.write("## Time-Series Trends (Monthly)")

    ts_choice = st.sidebar.radio(
        "Time-series dataset",
        [
            "Synthetic: 5-year baseline",
            "Synthetic: 5-year with economic shock",
            "Upload time-series CSV"
        ]
    )

    ts_uploaded = None
    if ts_choice == "Upload time-series CSV":
        ts_uploaded = st.sidebar.file_uploader(
            "Upload time-series CSV",
            type=["csv","xlsx"],
            help="Required: province, year, month, employees, avg_daily_wage, stress_score, burnout_score, absenteeism_score"
        )

    # Load timeseries data
    if ts_choice.startswith("Synthetic"):
        shock_flag = "shock" in ts_choice.lower()
        df_ts = generate_timeseries_data(years=5, shock=shock_flag, seed=999)
        source_label = "Synthetic with shock" if shock_flag else "Synthetic baseline"
    else:
        if ts_uploaded is None:
            st.info("Upload a time-series dataset to explore trends.")
            st.stop()
        try:
            df_ts = pd.read_csv(ts_uploaded) if ts_uploaded.name.endswith(".csv") else pd.read_excel(ts_uploaded)
            source_label = "User-uploaded time-series"
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

    st.write(f"**Current time-series source:** {source_label}")
    st.dataframe(df_ts.head(10))

    required_ts = [
        "province","year","month","employees","avg_daily_wage",
        "stress_score","burnout_score","absenteeism_score"
    ]
    miss_ts = [c for c in required_ts if c not in df_ts.columns]
    if miss_ts:
        st.error(f"Missing required time-series columns: {miss_ts}")
        st.stop()

    for c in ["employees","avg_daily_wage","stress_score","burnout_score","absenteeism_score","year","month"]:
        df_ts[c] = pd.to_numeric(df_ts[c], errors="coerce")

    df_ts_loss = compute_economic_loss(df_ts, WORKDAYS_PER_MONTH, BURNOUT_TO_PRESENTEEISM_SCALE)

    # Aggregate monthly national loss
    monthly = df_ts_loss.groupby(["year","month"]).agg(
        total_loss=("estimated_monthly_loss_cad","sum")
    ).reset_index()
    monthly["date_index"] = pd.to_datetime(
        dict(year=monthly["year"], month=monthly["month"], day=1)
    )

    st.write("### National monthly loss over time")
    fig2, ax2 = plt.subplots(figsize=(9,4))
    ax2.plot(monthly["date_index"], monthly["total_loss"])
    ax2.set_ylabel("Total loss (CAD)")
    ax2.set_xlabel("Month")
    st.pyplot(fig2)

    # Basic summary
    start_loss = monthly["total_loss"].iloc[0]
    end_loss = monthly["total_loss"].iloc[-1]
    change = end_loss - start_loss
    change_pct = (change / start_loss * 100) if start_loss != 0 else 0

    st.markdown(f"""
**Trend summary**  
- First month loss: **${start_loss:,.0f}**  
- Last month loss: **${end_loss:,.0f}**  
- Net change: **${change:,.0f}** (**{change_pct:.1f}%**)  

Time-series patterns can reveal **deterioration** (increasing loss) or **improvement** (declining loss), 
highlighting when mental-health risks are rising and when interventions may be working.
""")

# --------------------------------------------------------------------
# PAGE 3 – FORECASTING SANDBOX
# --------------------------------------------------------------------
elif page == "Forecasting Sandbox":
    st.write("## Forecasting Sandbox (MVP Predictive View)")

    fc_choice = st.sidebar.radio(
        "Forecasting dataset",
        [
            "Synthetic: 5-year baseline",
            "Synthetic: 5-year with economic shock",
            "Upload time-series CSV"
        ]
    )

    fc_uploaded = None
    if fc_choice == "Upload time-series CSV":
        fc_uploaded = st.sidebar.file_uploader(
            "Upload time-series CSV for forecasting",
            type=["csv","xlsx"],
            help="Required: province, year, month, employees, avg_daily_wage, stress_score, burnout_score, absenteeism_score"
        )

    # Load
    if fc_choice.startswith("Synthetic"):
        shock_flag = "shock" in fc_choice.lower()
        df_fc = generate_timeseries_data(years=5, shock=shock_flag, seed=1234)
        fc_label = "Synthetic with shock" if shock_flag else "Synthetic baseline"
    else:
        if fc_uploaded is None:
            st.info("Upload a time-series dataset to forecast.")
            st.stop()
        try:
            df_fc = pd.read_csv(fc_uploaded) if fc_uploaded.name.endswith(".csv") else pd.read_excel(fc_uploaded)
            fc_label = "User-uploaded"
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

    st.write(f"**Forecasting source:** {fc_label}")
    st.dataframe(df_fc.head(5))

    required_fc = [
        "province","year","month","employees","avg_daily_wage",
        "stress_score","burnout_score","absenteeism_score"
    ]
    miss_fc = [c for c in required_fc if c not in df_fc.columns]
    if miss_fc:
        st.error(f"Missing required columns: {miss_fc}")
        st.stop()

    for c in ["employees","avg_daily_wage","stress_score","burnout_score","absenteeism_score","year","month"]:
        df_fc[c] = pd.to_numeric(df_fc[c], errors="coerce")

    df_fc_loss = compute_economic_loss(df_fc, WORKDAYS_PER_MONTH, BURNOUT_TO_PRESENTEEISM_SCALE)

    monthly_fc = df_fc_loss.groupby(["year","month"]).agg(
        total_loss=("estimated_monthly_loss_cad","sum")
    ).reset_index()
    monthly_fc["date_index"] = pd.to_datetime(
        dict(year=monthly_fc["year"], month=monthly_fc["month"], day=1)
    )
    monthly_fc = monthly_fc.sort_values("date_index")

    # Simple linear trend forecast (MVP)
    y = monthly_fc["total_loss"].values
    x = np.arange(len(y))
    if len(y) >= 2:
        coeffs = np.polyfit(x, y, 1)
        trend = np.poly1d(coeffs)
        horizon = 12
        x_future = np.arange(len(y), len(y) + horizon)
        y_future = trend(x_future)

        last_date = monthly_fc["date_index"].iloc[-1]
        future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(), periods=horizon, freq="MS")

        hist_df = pd.DataFrame({"date": monthly_fc["date_index"], "total_loss": y})
        fut_df = pd.DataFrame({"date": future_dates, "total_loss": y_future})

        st.write("### Historical vs forecasted national loss (simple trend)")
        fig3, ax3 = plt.subplots(figsize=(9,4))
        ax3.plot(hist_df["date"], hist_df["total_loss"], label="Historical")
        ax3.plot(fut_df["date"], fut_df["total_loss"], linestyle="--", label="Forecast")
        ax3.set_ylabel("Total loss (CAD)")
        ax3.set_xlabel("Month")
        ax3.legend()
        st.pyplot(fig3)

        avg_future = y_future.mean()
        last_hist = y[-1]
        diff = avg_future - last_hist
        diff_pct = (diff / last_hist * 100) if last_hist != 0 else 0

        st.markdown(f"""
**Forecast interpretation (illustrative)**  
- Last historical month: **${last_hist:,.0f}** lost productivity  
- Average forecasted monthly loss (next 12 months): **${avg_future:,.0f}**  
- Change vs last observed month: **${diff:,.0f}** (**{diff_pct:.1f}%**)  

This simple trend model provides a **first-pass predictive view** of how losses might evolve.
A full implementation could incorporate richer models (e.g., time-series ML/AI) and scenario-based
simulation of different policy packages.
""")
    else:
        st.warning("Not enough time points for forecasting (need ≥ 2 months).")

# --------------------------------------------------------------------
# PAGE 4 – ABOUT & METHODOLOGY
# --------------------------------------------------------------------
elif page == "About & Methodology":
    st.write("## About & Methodology")

    st.markdown("""
### 1. Conceptual approach

LaborPulse-AI links **mental-health indicators** in SMEs (stress, burnout, absenteeism)
with **economic loss** to SMEs and the wider economy. It estimates:

- Lost days per employee (absenteeism + burnout-related presenteeism)
- Associated wage costs
- Aggregated loss by SME, province, and nationally

### 2. Data inputs

The model expects **aggregated SME data**, not individual-level records:

- SME ID, province, sector  
- Number of employees  
- Average daily wage  
- Stress, burnout, absenteeism scores (1–10)  

For time-series pages, additional fields are used:

- Year, month (for monthly series)  

### 3. Economic loss model (simple, interpretable)

1. Map scores to lost days per employee per month  
2. Multiply by average daily wage  
3. Scale by number of employees  
4. Aggregate across SMEs / provinces  

This yields **estimated monthly productivity loss (CAD)**, which can be compared
**before and after interventions**.

### 4. Interventions and scenarios

Interventions are simulated as **reductions in stress/burnout/absenteeism scores**.
The dashboard then recomputes:

- Loss per SME, province, national
- Cost reduction and percentage gains
- Reduced lost days per employee

### 5. Responsible AI principles

LaborPulse-AI:

- Works only with **aggregated data** (no worker profiling)
- Provides **transparent, interpretable metrics**
- Is intended to support **public-service and SME decision-making**, not HR sanctions
- Can be extended with **federated learning** and strong privacy guarantees in future phases

This MVP is intended as a **proof-of-concept** for funders, governments, and SME ecosystems,
showing how AI-enabled decision support can reduce mental-health–related productivity losses
and improve SME resilience.
""")

st.markdown('</div>', unsafe_allow_html=True)
