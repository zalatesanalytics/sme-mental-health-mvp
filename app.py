# app.py — LaborPulse-AI : Predictive Model to Improve SME Public Service Provision 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Page config & base styling
# -------------------------
st.set_page_config(
    page_title="LaborPulse-AI Model Predictive Model to Improve Public Service",
    layout="wide"
)

# Smarter background: soft gradient + white content block
st.markdown(
    """
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
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-block">', unsafe_allow_html=True)

# Header with small Canada flag + title
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/Flag_of_Canada.svg/320px-Flag_of_Canada.svg.png",
        use_column_width=True
    )
with col_title:
    st.title("LaborPulse-AI Model Predictive Model to Improve Public Service")
    st.caption(
        "Decision-support prototype for estimating SME workforce mental-health–related productivity losses "
        "in Canada and simulating the impact of public-service interventions."
    )

# About the AI decision-support agent
with st.expander("About LaborPulse-AI (AI decision-support agent)", expanded=False):
    st.markdown(
        """
        LaborPulse-AI is an **AI decision-support agent** designed to analyze aggregated workforce
        mental-health indicators from SMEs and evaluate how government or organizational interventions
        affect productivity and economic outcomes.

        **What this prototype does:**

        - **Impact visualization:** Shows how interventions change mental-health scores, lost days, and
          economic losses across provinces, using before/after comparisons and summary graphs.
        - **Predictive analytics (MVP):** Uses the current data and intervention scenario to estimate
          monthly and projected annual savings, providing a forward-looking view of potential gains.
        - **Outcome assessment:** Quantifies both **positive economic gains** (cost reductions) and
          **negative outcomes** (when an intervention increases losses), and reports them clearly.
        - **Decision-support role:** Only produces **organizational and policy-level outputs**; it never
          ranks or profiles individual workers and is intended to support responsible, evidence-based
          public-service decisions.

        Future versions can incorporate **continuous learning** from multiple historical datasets, and
        richer time-series models to forecast mental-health risks by province and nationally.
        """
    )

# -------------------------
# Synthetic data generator with scenario intensity
# -------------------------
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
        else:  # mixed
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

# -------------------------
# SIDEBAR: dataset scenario + inputs
# -------------------------
st.sidebar.title("LaborPulse-AI Controls")

scenario_choice = st.sidebar.radio(
    "Choose dataset scenario:",
    [
        "Upload my own CSV",
        "Sample: Low impact (healthier workforce)",
        "Sample: Medium impact",
        "Sample: High impact (severe mental-health strain)",
        "Sample: Mixed impact (varied SME risk)"
    ]
)

uploaded = None
if scenario_choice == "Upload my own CSV":
    uploaded = st.sidebar.file_uploader(
        "Upload SME CSV",
        type=["csv","xlsx"],
        help=(
            "Required columns: province, sme_id, sector, employees, avg_daily_wage, "
            "stress_score, burnout_score, absenteeism_score"
        )
    )
else:
    st.sidebar.info(
        "You selected a built-in sample dataset. "
        "Switch scenarios to see how losses and savings change under different SME risk profiles."
    )

# Economic model parameters
st.sidebar.subheader("Economic model parameters")
WORKDAYS_PER_MONTH = st.sidebar.number_input(
    "Workdays per month",
    value=20, min_value=15, max_value=25
)
PRESENTEEM_IMPACT = st.sidebar.number_input(
    "Presenteeism productivity loss factor (0–1)",
    value=0.30, min_value=0.0, max_value=1.0, step=0.01
)
BURNOUT_TO_PRESENTEEISM_SCALE = st.sidebar.number_input(
    "Burnout → presenteeism days scale (0–1)",
    value=0.10, min_value=0.0, max_value=1.0, step=0.01
)

# -------------------------
# Load data based on scenario
# -------------------------
def load_data():
    # 1) User-uploaded dataset
    if scenario_choice == "Upload my own CSV":
        if uploaded is None:
            st.info("Please upload an SME CSV file to proceed.")
            return None, "User-uploaded (waiting for file)"
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_local = pd.read_csv(uploaded)
            else:
                df_local = pd.read_excel(uploaded)
            return df_local, "User-uploaded dataset"
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return None, "Upload error"

    # 2) Synthetic scenarios
    if "Low impact" in scenario_choice:
        df_local = generate_synthetic_data(intensity="low", seed=10)
        label = "Sample: Low impact"
    elif "Medium impact" in scenario_choice:
        df_local = generate_synthetic_data(intensity="medium", seed=20)
        label = "Sample: Medium impact"
    elif "High impact" in scenario_choice:
        df_local = generate_synthetic_data(intensity="high", seed=30)
        label = "Sample: High impact"
    else:
        df_local = generate_synthetic_data(intensity="mixed", seed=40)
        label = "Sample: Mixed impact"
    return df_local, label

df, dataset_label = load_data()
if df is None:
    st.stop()

st.write(f"### Current dataset: {dataset_label}")
st.write("Preview of data (first 10 rows):")
st.dataframe(df.head(10))

required = [
    "province","sme_id","sector","employees","avg_daily_wage",
    "stress_score","burnout_score","absenteeism_score"
]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}. Please adjust your dataset.")
    st.stop()

# Ensure numeric types
for col in ["employees","avg_daily_wage","stress_score","burnout_score","absenteeism_score"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------------
# Economic loss calculation per SME
# -------------------------
def compute_economic_loss(df_in):
    df_l = df_in.copy()
    df_l['absenteeism_days_per_emp'] = df_l['absenteeism_score'].clip(1,10)
    df_l['presenteeism_days_equiv_per_emp'] = (
        (df_l['burnout_score'] / 10.0) * WORKDAYS_PER_MONTH * BURNOUT_TO_PRESENTEEISM_SCALE
    )
    df_l['lost_days_per_emp'] = df_l['absenteeism_days_per_emp'] + df_l['presenteeism_days_equiv_per_emp']
    df_l['loss_per_emp_cad'] = df_l['lost_days_per_emp'] * df_l['avg_daily_wage']
    df_l['estimated_monthly_loss_cad'] = df_l['loss_per_emp_cad'] * df_l['employees']
    df_l['estimated_monthly_loss_cad'] = df_l['estimated_monthly_loss_cad'].round(2)
    return df_l

df_losses = compute_economic_loss(df)

st.write("### Per-SME economic loss (sample rows)")
st.dataframe(
    df_losses[[
        "province","sme_id","sector","employees","avg_daily_wage",
        "stress_score","burnout_score","absenteeism_score",
        "lost_days_per_emp","estimated_monthly_loss_cad"
    ]].head(15)
)

# -------------------------
# Province-level aggregation (baseline)
# -------------------------
st.write("### Province-level aggregation (baseline)")
prov_agg = df_losses.groupby("province").agg(
    total_smes=("sme_id","nunique"),
    total_employees=("employees","sum"),
    avg_stress=("stress_score","mean"),
    avg_burnout=("burnout_score","mean"),
    avg_absenteeism=("absenteeism_score","mean"),
    avg_lost_days=("lost_days_per_emp","mean"),
    avg_daily_wage=("avg_daily_wage","mean"),
    estimated_monthly_loss_cad=("estimated_monthly_loss_cad","sum")
).reset_index()

prov_agg['estimated_monthly_loss_cad'] = prov_agg['estimated_monthly_loss_cad'].round(2)
st.dataframe(prov_agg.sort_values("estimated_monthly_loss_cad", ascending=False))

# High-level KPIs
total_loss = prov_agg['estimated_monthly_loss_cad'].sum()
total_employees = prov_agg['total_employees'].sum()
per_emp_loss = total_loss / total_employees if total_employees > 0 else 0.0

kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
with kpi_col1:
    st.metric("Baseline total monthly economic loss", f"${total_loss:,.0f}")
with kpi_col2:
    st.metric("Total employees covered", f"{int(total_employees):,}")
with kpi_col3:
    st.metric("Baseline loss per employee (monthly)", f"${per_emp_loss:,.2f}")

# -------------------------
# Visualization: top provinces by loss
# -------------------------
st.write("### Top provinces by estimated monthly loss (CAD)")
fig, ax = plt.subplots(figsize=(8,4))
prov_sorted = prov_agg.sort_values("estimated_monthly_loss_cad", ascending=True)
ax.barh(prov_sorted['province'], prov_sorted['estimated_monthly_loss_cad'])
ax.set_xlabel("Estimated monthly loss (CAD)")
st.pyplot(fig)

# -------------------------
# Intervention simulation — adjust scores and recompute losses
# -------------------------
st.write("### Scenario simulation: mental-health interventions")

intervention = st.selectbox(
    "Choose intervention type",
    ["None","Reduce stress scores","Reduce burnout scores","Reduce absenteeism scores"]
)
reduction_points = st.slider("Reduction points (1–5)", min_value=0, max_value=5, value=2)

def apply_intervention_and_recompute(df_in, intervention_type, points):
    df_sim = df_in.copy()
    if points > 0:
        if intervention_type == "Reduce stress scores":
            df_sim['stress_score'] = (df_sim['stress_score'] - points).clip(1,10)
        elif intervention_type == "Reduce burnout scores":
            df_sim['burnout_score'] = (df_sim['burnout_score'] - points).clip(1,10)
        elif intervention_type == "Reduce absenteeism scores":
            df_sim['absenteeism_score'] = (df_sim['absenteeism_score'] - points).clip(1,10)
    df_sim = compute_economic_loss(df_sim)
    prov_sim = df_sim.groupby("province").agg(
        total_smes=("sme_id","nunique"),
        total_employees=("employees","sum"),
        avg_stress=("stress_score","mean"),
        avg_burnout=("burnout_score","mean"),
        avg_absenteeism=("absenteeism_score","mean"),
        avg_lost_days=("lost_days_per_emp","mean"),
        estimated_monthly_loss_cad=("estimated_monthly_loss_cad","sum")
    ).reset_index()
    prov_sim['estimated_monthly_loss_cad'] = prov_sim['estimated_monthly_loss_cad'].round(2)
    return df_sim, prov_sim

df_sim, prov_sim = apply_intervention_and_recompute(df, intervention, reduction_points)

st.write(
    f"Projected province-level losses after **{intervention}** "
    f"(reduction = {reduction_points} points, dataset: {dataset_label}):"
)
st.dataframe(prov_sim.sort_values("estimated_monthly_loss_cad", ascending=False))

baseline_total = prov_agg['estimated_monthly_loss_cad'].sum()
sim_total = prov_sim['estimated_monthly_loss_cad'].sum()
savings = baseline_total - sim_total
delta_pct = (savings / baseline_total * 100) if baseline_total > 0 else 0.0

# Outcome assessment: gains vs losses
overall_label = "Monthly economic gain (cost reduction)"
if savings < 0:
    overall_label = "Monthly economic loss (intervention increases costs)"

st.metric(
    overall_label,
    f"${savings:,.0f}",
    delta=f"{delta_pct:.1f}%"
)

# Simple predictive outlook (extrapolation)
projected_annual_savings = savings * 12
st.caption(
    f"If this intervention level is sustained for 12 months, the projected annual impact is "
    f"**{'saving' if projected_annual_savings >= 0 else 'additional cost of'} "
    f"${abs(projected_annual_savings):,.0f}** (assuming no other shocks)."
)

# -------------------------
# Impact of intervention by province
# -------------------------
st.write("### Impact of intervention by province")

impact_df = prov_agg.merge(
    prov_sim,
    on="province",
    suffixes=("_baseline", "_after")
)

impact_df["monthly_savings_cad"] = (
    impact_df["estimated_monthly_loss_cad_baseline"] -
    impact_df["estimated_monthly_loss_cad_after"]
)
impact_df["savings_pct"] = np.where(
    impact_df["estimated_monthly_loss_cad_baseline"] > 0,
    100 * impact_df["monthly_savings_cad"] / impact_df["estimated_monthly_loss_cad_baseline"],
    0.0
)
impact_df["efficiency_gain_cad_per_employee"] = np.where(
    impact_df["total_employees_baseline"] > 0,
    impact_df["monthly_savings_cad"] / impact_df["total_employees_baseline"],
    0.0
)
impact_df["reduction_lost_days_per_emp"] = (
    impact_df["avg_lost_days_baseline"] - impact_df["avg_lost_days_after"]
)

impact_view = impact_df[[
    "province",
    "estimated_monthly_loss_cad_baseline",
    "estimated_monthly_loss_cad_after",
    "monthly_savings_cad",
    "savings_pct",
    "avg_lost_days_baseline",
    "avg_lost_days_after",
    "reduction_lost_days_per_emp",
    "efficiency_gain_cad_per_employee"
]].sort_values("monthly_savings_cad", ascending=False)

impact_view = impact_view.rename(columns={
    "estimated_monthly_loss_cad_baseline": "loss_baseline_cad",
    "estimated_monthly_loss_cad_after": "loss_after_cad",
    "monthly_savings_cad": "cost_reduction_cad",
    "savings_pct": "cost_reduction_pct",
    "avg_lost_days_baseline": "lost_days_per_emp_baseline",
    "avg_lost_days_after": "lost_days_per_emp_after"
})

st.dataframe(impact_view.style.format({
    "loss_baseline_cad": "{:,.0f}",
    "loss_after_cad": "{:,.0f}",
    "cost_reduction_cad": "{:,.0f}",
    "cost_reduction_pct": "{:,.1f}",
    "lost_days_per_emp_baseline": "{:,.2f}",
    "lost_days_per_emp_after": "{:,.2f}",
    "reduction_lost_days_per_emp": "{:,.2f}",
    "efficiency_gain_cad_per_employee": "{:,.2f}"
}))

st.markdown(
    """
    **Interpretation (policy-level):**

    - **Cost reduction (CAD & %)** – `cost_reduction_cad` and `cost_reduction_pct` capture the
      estimated reduction in mental-health–related productivity loss per month in each province.
      Positive values signal **economic gains**; negative values flag **added costs**.
    - **Efficiency gain (CAD/employee)** – `efficiency_gain_cad_per_employee` approximates cost
      savings per worker per month. Higher values suggest improved workforce efficiency and better
      use of public or employer resources.
    - **Reduction in lost days/employee** – `reduction_lost_days_per_emp` shows fewer days lost
      to absenteeism and burnout per employee. This acts as a proxy for **faster response**, better
      access to services, and reduced strain on frontline health and social systems.
    - **Provincial comparison** – provinces with higher baseline losses and larger cost reductions
      may be priority targets for scaling up support (e.g., mental-health benefits, grants, or
      psychosocial programs).
    - **Outcome assessment** – when **cost_reduction_cad > 0**, the intervention generates
      measurable productivity and economic gains; when it is **< 0**, the model clearly reports
      a net loss so policymakers can reconsider or redesign the intervention.

    LaborPulse-AI only works with **aggregated SME and provincial-level data**. It is not designed
    to profile or score individual workers, and all insights are intended to support responsible,
    transparent public-service decisions.
    """
)

# -------------------------
# Export reports
# -------------------------
st.write("### Export scenario results")
csv_bytes_prov = prov_sim.to_csv(index=False).encode('utf-8')
st.download_button(
    "Download simulated province-level report (CSV)",
    csv_bytes_prov,
    file_name="laborpulse_prov_simulated_losses.csv",
    mime="text/csv"
)

csv_bytes_sme = df_sim.to_csv(index=False).encode('utf-8')
st.download_button(
    "Download simulated SME-level report (CSV)",
    csv_bytes_sme,
    file_name="laborpulse_sme_simulated_losses.csv",
    mime="text/csv"
)

st.write("---")
st.caption(
    "LaborPulse-AI is a prototype decision-support tool. Values are based on synthetic or aggregated SME data "
    "and configurable assumptions. For real-world policy, calibrate parameters with administrative and payroll data, "
    "and consider integrating multi-year time-series for richer forecasting."
)

st.markdown('</div>', unsafe_allow_html=True)
