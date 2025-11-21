# app.py — SME Mental Health Decision Support (with Economic Loss)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="SME Mental Health Decision Support", layout="wide")
st.title("AI Decision Support — SME Mental Health & Economic Impact (with Economic Loss)")

# -------------------------
# Synthetic data generator (1-10 scores + employees + wage)
# -------------------------
@st.cache_data
def generate_synthetic_data(seed=42, n_provinces=10, smes_per_prov=(8,15)):
    np.random.seed(seed)
    provinces = [
        'Ontario','Quebec','British Columbia','Alberta','Manitoba',
        'Saskatchewan','Nova Scotia','New Brunswick',
        'Newfoundland & Labrador','Prince Edward Island'
    ][:n_provinces]
    sectors = ['Tech','Retail','Manufacturing','Services']
    rows = []
    for prov in provinces:
        num_smes = np.random.randint(smes_per_prov[0], smes_per_prov[1]+1)
        for i in range(num_smes):
            employees = int(np.random.randint(5,200))  # employees per SME
            avg_daily_wage = float(np.round(np.random.uniform(100,400),2))  # CAD/day
            # Scores 1-10 (integers)
            stress = int(np.random.randint(1,11))
            burnout = int(np.random.randint(1,11))
            absenteeism = int(np.random.randint(1,11))  # interpret as absentee days per month (1-10)
            # Build row
            rows.append({
                "province": prov,
                "sme_id": f"{prov[:3].upper()}_{i+1}",
                "sector": np.random.choice(sectors),
                "employees": employees,
                "avg_daily_wage": avg_daily_wage,
                "stress_score": stress,
                "burnout_score": burnout,
                "absenteeism_score": absenteeism
            })
    return pd.DataFrame(rows)

# -------------------------
# Data input (upload or synthetic)
# -------------------------
st.sidebar.header("Data input")
uploaded = st.sidebar.file_uploader("Upload SME CSV (optional)", type=["csv","xlsx"])
load_sample = st.sidebar.button("Load synthetic sample data")

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.sidebar.success(f"Loaded {uploaded.name} ({df.shape[0]} rows)")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        st.stop()
elif load_sample:
    df = generate_synthetic_data()
    st.sidebar.success("Synthetic sample data loaded.")
else:
    st.info("Upload SME CSV or click 'Load synthetic sample data' in the sidebar to start.")
    st.stop()

# -------------------------
# Validate required columns; if missing, warn and stop
# -------------------------
required = ["province","sme_id","employees","avg_daily_wage","absenteeism_score","burnout_score","stress_score"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.warning(f"Missing required columns: {missing}. The synthetic dataset contains them.")
    st.write("Current columns:", list(df.columns))
    st.stop()

# Ensure numeric types
for c in ["employees","avg_daily_wage","absenteeism_score","burnout_score","stress_score"]:
    df[c] = pd.to_numeric(df[c], errors='coerce')

st.write("### Data preview")
st.dataframe(df.head(10))

# -------------------------
# Economic loss model parameters
# -------------------------
st.write("### Economic loss model parameters")
col1, col2, col3 = st.columns(3)
with col1:
    WORKDAYS_PER_MONTH = st.number_input("Workdays per month", value=20, min_value=15, max_value=25)
with col2:
    PRESENTEEM_IMPACT = st.number_input("Presenteeism productivity loss factor (0-1)", value=0.30, min_value=0.0, max_value=1.0, step=0.01)
with col3:
    BURNOUT_TO_PRESENTEEISM_SCALE = st.number_input("Burnout → presenteeism days scale (0-1)", value=0.10, min_value=0.0, max_value=1.0, step=0.01)

st.caption("Interpretation: absenteeism_score (1-10) ≈ absentee days/month per employee. Burnout_score (1-10) contributes to additional presenteeism-equivalent days (scaled).")

# -------------------------
# Economic loss calculation per SME
# -------------------------
def compute_economic_loss(df):
    df = df.copy()
    df['absenteeism_days_per_emp'] = df['absenteeism_score'].clip(1,10)
    df['presenteeism_days_equiv_per_emp'] = (df['burnout_score'] / 10.0) * WORKDAYS_PER_MONTH * BURNOUT_TO_PRESENTEEISM_SCALE
    df['lost_days_per_emp'] = df['absenteeism_days_per_emp'] + df['presenteeism_days_equiv_per_emp']
    df['loss_per_emp_cad'] = df['lost_days_per_emp'] * df['avg_daily_wage']
    df['estimated_monthly_loss_cad'] = df['loss_per_emp_cad'] * df['employees']
    df['estimated_monthly_loss_cad'] = df['estimated_monthly_loss_cad'].round(2)
    return df

df_losses = compute_economic_loss(df)
st.write("### Per-SME economic loss (sample rows)")
st.dataframe(
    df_losses[[
        "province","sme_id","employees","avg_daily_wage",
        "absenteeism_score","burnout_score","stress_score",
        "lost_days_per_emp","estimated_monthly_loss_cad"
    ]].head(15)
)

# -------------------------
# Aggregate by province
# -------------------------
st.write("### Province-level aggregation")
prov_agg = df_losses.groupby("province").agg(
    total_smes = ("sme_id","nunique"),
    total_employees = ("employees","sum"),
    avg_absenteeism = ("absenteeism_score","mean"),
    avg_burnout = ("burnout_score","mean"),
    avg_stress = ("stress_score","mean"),
    avg_daily_wage = ("avg_daily_wage","mean"),
    estimated_monthly_loss_cad = ("estimated_monthly_loss_cad","sum")
).reset_index()

prov_agg['estimated_monthly_loss_cad'] = prov_agg['estimated_monthly_loss_cad'].round(2)
st.dataframe(prov_agg.sort_values("estimated_monthly_loss_cad", ascending=False))

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
st.write("### Scenario simulation: interventions")
intervention = st.selectbox("Choose intervention", ["None","Reduce stress by N points","Reduce burnout by N points","Reduce absenteeism by N points"])
reduction_points = st.slider("Reduction points (1-5)", min_value=0, max_value=5, value=2)

def apply_intervention_and_recompute(df, intervention, points):
    df_sim = df.copy()
    if intervention == "Reduce stress by N points":
        df_sim['stress_score'] = (df_sim['stress_score'] - points).clip(1,10)
    elif intervention == "Reduce burnout by N points":
        df_sim['burnout_score'] = (df_sim['burnout_score'] - points).clip(1,10)
    elif intervention == "Reduce absenteeism by N points":
        df_sim['absenteeism_score'] = (df_sim['absenteeism_score'] - points).clip(1,10)
    df_sim = compute_economic_loss(df_sim)
    prov_sim = df_sim.groupby("province").agg(
        total_smes = ("sme_id","nunique"),
        total_employees = ("employees","sum"),
        avg_absenteeism = ("absenteeism_score","mean"),
        avg_burnout = ("burnout_score","mean"),
        avg_stress = ("stress_score","mean"),
        estimated_monthly_loss_cad = ("estimated_monthly_loss_cad","sum")
    ).reset_index()
    prov_sim['estimated_monthly_loss_cad'] = prov_sim['estimated_monthly_loss_cad'].round(2)
    return df_sim, prov_sim

df_sim, prov_sim = apply_intervention_and_recompute(df, intervention, reduction_points)

st.write(f"Projected province-level losses after **{intervention}** (reduction = {reduction_points} points):")
st.dataframe(prov_sim.sort_values("estimated_monthly_loss_cad", ascending=False))

baseline_total = prov_agg['estimated_monthly_loss_cad'].sum()
sim_total = prov_sim['estimated_monthly_loss_cad'].sum()
savings = baseline_total - sim_total
delta_pct = (savings / baseline_total * 100) if baseline_total > 0 else 0.0
st.metric("Estimated total monthly savings (all provinces)", f"${savings:,.2f}", delta=f"{delta_pct:.1f}%")

# -------------------------
# Export aggregated reports (CSV)
# -------------------------
st.write("### Export")
csv_bytes = prov_sim.to_csv(index=False).encode('utf-8')
st.download_button(
    "Download simulated province-level report (CSV)",
    csv_bytes,
    file_name="prov_simulated_losses.csv",
    mime="text/csv"
)

csv_bytes_sme = df_sim.to_csv(index=False).encode('utf-8')
st.download_button(
    "Download simulated SME-level report (CSV)",
    csv_bytes_sme,
    file_name="sme_simulated_losses.csv",
    mime="text/csv"
)

st.write("---")
st.caption(
    "This MVP uses 1–10 scores for absenteeism, burnout, and stress to estimate lost workdays "
    "and economic loss. Parameters can be calibrated with real SME and labour data during a pilot."
)
