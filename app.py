# app.py — LaborPulse-AI 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Try to load scikit-learn for the AI model
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# -------------------------
# Page config & base styling
# -------------------------
st.set_page_config(
    page_title="LaborPulse-AI To Improve Public Service",
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
    @media (max-width: 768px) {
        .main-block {
            padding: 1rem;
        }
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
    st.title("LaborPulse-AI To Improve Public Service")
    st.caption(
        "Decision-support prototype for estimating SME workforce mental-health–related productivity losses "
        "in Canada, forecasting risks, and simulating the impact of public-service interventions."
    )

# --------------------------------------------------------------------
# ABOUT AGENT (GLOBAL) — ALWAYS VISIBLE
# --------------------------------------------------------------------
st.markdown("""
### About LaborPulse-AI (AI decision-support agent)

LaborPulse-AI analyzes **aggregated** SME workforce mental-health indicators to estimate
economic loss, simulate interventions, and support **policy-level** and **organizational**
decisions – never individual profiling.

- **Impact visualization:** dashboards for loss, savings, and lost days  
- **AI learning & prediction:** trains a model on available data to predict losses  
- **12-month forecasting:** projects mental-health risk indicators and at-risk employees by province  
- **Outcome assessment:** highlights both **gains** (savings) and **losses** (when an intervention backfires)  
- **Responsible AI:** only aggregated SME/provincial outputs, no worker-level scoring  

In this version, **stress scores directly increase lost productive days** via a stress-related
presenteeism factor. Reducing stress or burnout now leads to visible **economic savings**.
""")

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

# Default is "Sample: Medium impact" (index=2)
scenario_choice = st.sidebar.radio(
    "Choose dataset scenario:",
    [
        "Upload my own CSV",
        "Sample: Low impact (healthier workforce)",
        "Sample: Medium impact",
        "Sample: High impact (severe mental-health strain)",
        "Sample: Mixed impact (varied SME risk)"
    ],
    index=2
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
        "You are using a built-in **Medium impact** sample by default. "
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
STRESS_TO_PRESENTEEISM_SCALE = st.sidebar.number_input(
    "Stress → presenteeism days scale (0–1)",
    value=0.05, min_value=0.0, max_value=1.0, step=0.01,
    help="How much stress (1–10) converts into extra 'lost days' via reduced focus, fatigue, etc."
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
st.write("Preview of data (first 5 rows):")
st.dataframe(df.head(5), use_container_width=True)

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
# Economic loss calculation per SME (rule-based with STRESS EFFECT)
# -------------------------
def compute_economic_loss(df_in):
    df_l = df_in.copy()

    # 1) Absenteeism: direct days lost
    df_l['absenteeism_days_per_emp'] = df_l['absenteeism_score'].clip(1,10)

    # 2) Burnout-related presenteeism: reduced productivity while present
    df_l['presenteeism_days_equiv_per_emp'] = (
        (df_l['burnout_score'] / 10.0) * WORKDAYS_PER_MONTH * BURNOUT_TO_PRESENTEEISM_SCALE
    )

    # 3) NEW: Stress-related presenteeism: cognitive load, fatigue, lower focus
    df_l['stress_days_equiv_per_emp'] = (
        (df_l['stress_score'] / 10.0) * WORKDAYS_PER_MONTH * STRESS_TO_PRESENTEEISM_SCALE
    )

    # 4) Total lost days per employee
    df_l['lost_days_per_emp'] = (
        df_l['absenteeism_days_per_emp']
        + df_l['presenteeism_days_equiv_per_emp']
        + df_l['stress_days_equiv_per_emp']
    )

    # 5) Monetary loss
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
        "stress_days_equiv_per_emp","presenteeism_days_equiv_per_emp",
        "lost_days_per_emp","estimated_monthly_loss_cad"
    ]].head(5),
    use_container_width=True
)

# --------------------------------------------------------------------
# AI LEARNING & PREDICTION SECTION
# --------------------------------------------------------------------
st.write("## AI Learning & Prediction (Random Forest)")

ai_model = None
ai_r2 = None
ai_rmse = None
ai_feature_importance = None
ai_baseline_total = None

FEATURES = ["stress_score", "burnout_score", "absenteeism_score", "employees", "avg_daily_wage"]

if SKLEARN_AVAILABLE:
    # Drop rows with missing values in the feature set or target
    df_train = df_losses.dropna(subset=FEATURES + ["estimated_monthly_loss_cad"]).copy()
    if len(df_train) >= 20:  # need enough rows to train something reasonable
        X = df_train[FEATURES]
        y = df_train["estimated_monthly_loss_cad"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        ai_model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        ai_model.fit(X_train, y_train)

        y_pred = ai_model.predict(X_test)
        ai_r2 = r2_score(y_test, y_pred)
        ai_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Feature importance
        importances = ai_model.feature_importances_
        ai_feature_importance = (
            pd.DataFrame({
                "feature": FEATURES,
                "importance": importances
            })
            .sort_values("importance", ascending=False)
        )

        # AI-estimated baseline total monthly loss
        ai_baseline_total = ai_model.predict(df_losses[FEATURES]).sum()

        k1, k2 = st.columns(2)
        with k1:
            st.metric("AI model R² (prediction power)", f"{ai_r2:.2f}")
        with k2:
            st.metric("AI model RMSE (CAD)", f"${ai_rmse:,.0f}")

        st.write("Top drivers of predicted economic loss (feature importance):")
        st.dataframe(ai_feature_importance, use_container_width=True)

    else:
        st.warning("Not enough rows to train the AI model (need at least 20).")
else:
    st.warning(
        "scikit-learn is not installed. To enable AI learning, add `scikit-learn` "
        "to your requirements.txt and redeploy."
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
st.dataframe(
    prov_agg.sort_values("estimated_monthly_loss_cad", ascending=False),
    use_container_width=True
)

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
fig, ax = plt.subplots(figsize=(6,4))
prov_sorted = prov_agg.sort_values("estimated_monthly_loss_cad", ascending=True)
ax.barh(prov_sorted['province'], prov_sorted['estimated_monthly_loss_cad'])
ax.set_xlabel("Estimated monthly loss (CAD)")
st.pyplot(fig, use_container_width=True)

# -------------------------
# Scenario simulation — adjust scores and recompute losses
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
st.dataframe(
    prov_sim.sort_values("estimated_monthly_loss_cad", ascending=False),
    use_container_width=True
)

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

# --- AI-estimated savings for this scenario (using learned model) ---
ai_sim_total = None
ai_scenario_savings = None
if 'ai_model' in globals() and ai_model is not None:
    try:
        ai_sim_total = ai_model.predict(df_sim[FEATURES]).sum()
        ai_scenario_savings = ai_baseline_total - ai_sim_total if ai_baseline_total is not None else None
        st.caption(
            f"AI-estimated monthly loss (baseline): **${ai_baseline_total:,.0f}** | "
            f"AI-estimated monthly loss (scenario): **${ai_sim_total:,.0f}** | "
            f"AI-estimated savings: **${ai_scenario_savings:,.0f}**"
        )
    except Exception:
        st.caption("AI model could not compute scenario estimates for this dataset.")

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

st.dataframe(
    impact_view.style.format({
        "loss_baseline_cad": "{:,.0f}",
        "loss_after_cad": "{:,.0f}",
        "cost_reduction_cad": "{:,.0f}",
        "cost_reduction_pct": "{:,.1f}",
        "lost_days_per_emp_baseline": "{:,.2f}",
        "lost_days_per_emp_after": "{:,.2f}",
        "reduction_lost_days_per_emp": "{:,.2f}",
        "efficiency_gain_cad_per_employee": "{:,.2f}"
    }),
    use_container_width=True
)

# --- Data-driven interpretation (policy-level) ---

# National weighted averages for lost days (before & after)
if total_employees > 0:
    nat_lost_before = (
        (prov_agg["avg_lost_days"] * prov_agg["total_employees"]).sum() / total_employees
    )
    nat_lost_after = (
        (prov_sim["avg_lost_days"] * prov_sim["total_employees"]).sum() / total_employees
    )
else:
    nat_lost_before = nat_lost_after = 0.0

# Top province by absolute cost reduction (can be negative if costs increase)
top_row = impact_view.iloc[0] if not impact_view.empty else None
if top_row is not None:
    top_province = top_row["province"]
    top_loss_before = top_row["loss_baseline_cad"]
    top_loss_after = top_row["loss_after_cad"]
    top_saving = top_row["cost_reduction_cad"]
    top_pct = top_row["cost_reduction_pct"]
else:
    top_province = None
    top_loss_before = top_loss_after = top_saving = top_pct = 0.0

# Weighted average efficiency gain per employee across provinces
if total_employees > 0:
    avg_eff_gain_per_emp = (
        impact_df["efficiency_gain_cad_per_employee"]
        * impact_df["total_employees_baseline"]
    ).sum() / total_employees
else:
    avg_eff_gain_per_emp = 0.0

direction = "reduction" if savings >= 0 else "increase"
direction_word = "gain" if savings >= 0 else "loss"

st.markdown(
    f"""
    **Interpretation for this scenario (policy-level)**  
    _Dataset: **{dataset_label}**, Intervention: **{intervention}**, Reduction: **{reduction_points} points**_

    - Across all participating SMEs (**{int(total_employees):,} workers**), baseline mental-health–related
      productivity loss is estimated at **${baseline_total:,.0f} per month**. Under the selected intervention,
      losses change to **${sim_total:,.0f} per month**, a net **{direction} of ${abs(savings):,.0f}**
      (**{delta_pct:.1f}% {direction_word}**).
    - At the national level, average lost days per employee shift from
      **{nat_lost_before:.2f} days/month** to **{nat_lost_after:.2f} days/month**. This indicates
      a **{('decline' if nat_lost_after < nat_lost_before else 'rise')} in time lost to absenteeism, burnout, and stress**, 
      which is a proxy for **response time to mental-health needs** and overall workplace support.
    """
)

if top_province is not None:
    st.markdown(
        f"""
        - The province with the largest absolute cost impact is **{top_province}**, where monthly
          mental-health–related productivity losses change from **${top_loss_before:,.0f}** to
          **${top_loss_after:,.0f}**, a net **{('reduction' if top_saving >= 0 else 'increase')} of ${abs(top_saving):,.0f}**
          (**{top_pct:.1f}%**). This suggests **{top_province}** is a priority candidate for 
          scaling up or redesigning interventions.
        """
    )

st.markdown(
    f"""
    - On average, the intervention generates an estimated **efficiency gain of ${avg_eff_gain_per_emp:,.2f}
      per employee per month**. This can be interpreted as improved productivity and better use of public or
      employer resources when investments in mental-health support are effective.
    - Provinces with higher **baseline losses** and larger **cost reductions** are strong candidates for
      targeted public-service investment (e.g., mental-health benefits, grants, training, or psychosocial programs).
    - When **cost_reduction_cad > 0**, the intervention produces a **positive economic return** through lower 
      mental-health–related productivity loss. When **cost_reduction_cad < 0**, LaborPulse-AI clearly reports
      a **net loss**, signaling that policymakers may need to **adjust, retarget, or redesign** the intervention.
    - All estimates are based on **aggregated SME and provincial-level data** only; LaborPulse-AI does **not**
      score or profile individual workers. It is designed as a **responsible AI decision-support tool** to 
      inform **organizational and policy-level actions**, not individual HR decisions.
    """
)

# --------------------------------------------------------------------
# AI-BASED RECOMMENDED INTERVENTION LIST (FOR GOVERNMENT)
# --------------------------------------------------------------------
st.write("### AI-based recommended intervention priorities for government")

if 'ai_model' in globals() and ai_model is not None and ai_baseline_total is not None:
    def ai_simulate_intervention(df_base, intervention_type, points):
        df_tmp = df_base.copy()
        if points > 0:
            if intervention_type == "Reduce stress scores":
                df_tmp["stress_score"] = (df_tmp["stress_score"] - points).clip(1,10)
            elif intervention_type == "Reduce burnout scores":
                df_tmp["burnout_score"] = (df_tmp["burnout_score"] - points).clip(1,10)
            elif intervention_type == "Reduce absenteeism scores":
                df_tmp["absenteeism_score"] = (df_tmp["absenteeism_score"] - points).clip(1,10)
        df_tmp = compute_economic_loss(df_tmp)  # keep consistent structure
        total_pred = ai_model.predict(df_tmp[FEATURES]).sum()
        return total_pred

    test_points = 2  # modest but realistic scenario
    scenarios = [
        "Reduce burnout scores",
        "Reduce stress scores",
        "Reduce absenteeism scores"
    ]
    rows = []
    for s in scenarios:
        ai_total = ai_simulate_intervention(df, s, test_points)
        rows.append({
            "intervention_type": s,
            "reduction_points_tested": test_points,
            "ai_estimated_monthly_savings_cad": ai_baseline_total - ai_total
        })
    rec_df = pd.DataFrame(rows).sort_values(
        "ai_estimated_monthly_savings_cad", ascending=False
    )
    st.dataframe(
        rec_df.style.format({
            "ai_estimated_monthly_savings_cad": "{:,.0f}"
        }),
        use_container_width=True
    )

    st.markdown(
        """
        **Interpretation (AI-based recommendations):**

        - The table ranks intervention types by **AI-estimated monthly savings**, assuming a modest
          reduction of 2 points in the relevant risk scores.
        - Interventions at the top (often **burnout- or stress-focused** in many SME contexts) tend to produce
          the largest economic returns and may warrant **priority investment** (e.g., counseling benefits,
          manager training, psychosocial support).
        - This ranking is **data-driven** and updates automatically when you upload new SME datasets or
          adjust model parameters, illustrating how LaborPulse-AI can **continuously learn** from new data
          to refine policy guidance.
        """
    )
else:
    st.caption(
        "AI-based ranking is unavailable because the AI model could not be trained "
        "(either scikit-learn is missing or there were too few rows)."
    )

# --------------------------------------------------------------------
# 12-MONTH FORECAST: PROVINCIAL & NATIONAL MENTAL-HEALTH OUTLOOK
# --------------------------------------------------------------------
st.write("## 12-Month Forecast: Provincial Mental-Health Trends & At-Risk Employees")

@st.cache_data
def build_12m_forecast(prov_table, intervention_type, points):
    """
    Scenario-based 12-month forecast using current (post-intervention) provincial averages
    as the starting point. This is a simple, interpretable projection:
    - With no intervention: slight upward drift in risk.
    - With interventions: gentle downward drift in stress/burnout/absenteeism.
    """
    rng = np.random.default_rng(123)
    months = pd.date_range(pd.Timestamp.today().normalize(), periods=12, freq="MS")

    records = []

    for _, row in prov_table.iterrows():
        prov = row["province"]
        base_s = row["avg_stress"]
        base_b = row["avg_burnout"]
        base_a = row["avg_absenteeism"]
        emp = row["total_employees"]

        # Direction of trend based on intervention
        if intervention_type == "None" or points == 0:
            # Slight worsening over time without action
            s_slope = 0.04
            b_slope = 0.05
            a_slope = 0.03
            slope_scale = 1.0
        else:
            # Interventions push risks downward
            scale = points / 5.0  # 0–1
            s_slope = -0.10 * scale
            b_slope = -0.12 * scale
            a_slope = -0.08 * scale
            slope_scale = 1.0

        for i, dt in enumerate(months, start=1):
            # Month fraction (0–1)
            t = i

            # Add small noise for realism
            eps_s = rng.normal(0, 0.10)
            eps_b = rng.normal(0, 0.10)
            eps_a = rng.normal(0, 0.10)

            stress_t = np.clip(base_s + s_slope * t + eps_s, 1, 10)
            burnout_t = np.clip(base_b + b_slope * t + eps_b, 1, 10)
            abs_t = np.clip(base_a + a_slope * t + eps_a, 1, 10)

            composite = (stress_t + burnout_t + abs_t) / 3.0

            # Probability that an employee experiences at least one major challenge
            # (stress/anxiety/depression proxy) in that month.
            prob_challenge = np.clip(
                0.05 + 0.05 * (composite / 10.0) + 0.02 * (abs_t / 10.0),
                0.0,
                0.80
            )
            expected_at_risk = emp * prob_challenge

            records.append({
                "province": prov,
                "month_index": i,
                "month_date": dt,
                "month_label": dt.strftime("%Y-%m"),
                "forecast_stress": stress_t,
                "forecast_burnout": burnout_t,
                "forecast_absenteeism": abs_t,
                "composite_index": composite,
                "prob_challenge": prob_challenge,
                "employees": emp,
                "expected_employees_at_risk": expected_at_risk
            })

    return pd.DataFrame(records)

# Forecast is based on the *post-intervention* provincial averages (prov_sim)
forecast_df = build_12m_forecast(prov_sim, intervention, reduction_points)

# --- Provincial trend visualization ---
prov_list = sorted(forecast_df["province"].unique())
selected_prov = st.selectbox("Select province to view 12-month forecast", prov_list)

prov_forecast = forecast_df[forecast_df["province"] == selected_prov].copy()

st.write(f"### 12-month forecast for {selected_prov} (scenario: {intervention}, reduction = {reduction_points} points)")
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(prov_forecast["month_label"], prov_forecast["forecast_stress"], marker="o", label="Stress")
ax.plot(prov_forecast["month_label"], prov_forecast["forecast_burnout"], marker="o", label="Burnout")
ax.plot(prov_forecast["month_label"], prov_forecast["forecast_absenteeism"], marker="o", label="Absenteeism")
ax.set_ylabel("Risk score (1–10)")
ax.set_xlabel("Month")
ax.set_title(f"Forecasted mental-health indicators – {selected_prov}")
ax.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)

st.write("Forecast table (province-level, next 12 months):")
st.dataframe(
    prov_forecast[[
        "month_label",
        "forecast_stress",
        "forecast_burnout",
        "forecast_absenteeism",
        "composite_index",
        "prob_challenge",
        "expected_employees_at_risk"
    ]],
    use_container_width=True
)

# --- National forecast summary ---
national_month = forecast_df.groupby("month_label").agg(
    total_employees=("employees", "sum"),
    expected_at_risk=("expected_employees_at_risk", "sum"),
    avg_composite=("composite_index", "mean")
).reset_index()

national_month["percent_at_risk"] = np.where(
    national_month["total_employees"] > 0,
    100.0 * national_month["expected_at_risk"] / national_month["total_employees"],
    0.0
)

st.write("### National forecast summary (next 12 months)")
st.dataframe(
    national_month[[
        "month_label",
        "avg_composite",
        "expected_at_risk",
        "total_employees",
        "percent_at_risk"
    ]],
    use_container_width=True
)

fig2, ax2 = plt.subplots(figsize=(7,4))
ax2.plot(
    national_month["month_label"],
    national_month["percent_at_risk"],
    marker="o"
)
ax2.set_ylabel("Employees at risk (%)")
ax2.set_xlabel("Month")
ax2.set_title("National forecast: % of employees at risk of major mental-health challenge")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig2, use_container_width=True)

# --- Provincial expected at-risk employees (average month over next 12) ---
prov_risk = forecast_df.groupby("province").agg(
    total_employees=("employees", "max"),
    avg_prob_challenge=("prob_challenge", "mean"),
    expected_at_risk_per_month=("expected_employees_at_risk", "mean")
).reset_index()

prov_risk["percent_at_risk_per_month"] = np.where(
    prov_risk["total_employees"] > 0,
    100.0 * prov_risk["expected_at_risk_per_month"] / prov_risk["total_employees"],
    0.0
)

st.write("### Expected employees at risk (per month, next 12 months, by province)")
st.dataframe(
    prov_risk.sort_values("expected_at_risk_per_month", ascending=False).style.format({
        "avg_prob_challenge": "{:.2f}",
        "expected_at_risk_per_month": "{:,.0f}",
        "percent_at_risk_per_month": "{:.1f}"
    }),
    use_container_width=True
)

st.markdown(
    """
    **Interpretation (forecasted mental-health challenges):**

    - For each province, the forecasted **stress, burnout, and absenteeism scores** are projected
      over the next 12 months based on the current scenario (including any selected intervention).
    - Using a simple, interpretable risk model, LaborPulse-AI estimates the probability that a worker
      will experience at least one major mental-health challenge (stress/anxiety/depression proxy)
      in a given month.
    - The **national forecast summary** shows how many employees, in absolute numbers and as a
      percentage of the workforce, are expected to be at risk each month.
    - The **province-level at-risk table** helps identify jurisdictions where the **burden of mental-health
      risk is highest**, supporting prioritization of public investments and policy responses.
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

csv_bytes_forecast = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download 12-month forecast (province × month, CSV)",
    csv_bytes_forecast,
    file_name="laborpulse_12m_forecast.csv",
    mime="text/csv"
)

st.write("---")
st.caption(
    "LaborPulse-AI is a prototype decision-support tool. Values are based on synthetic or aggregated SME data "
    "and configurable assumptions. For real-world policy, calibrate parameters with administrative and payroll data, "
    "and integrate multi-year time-series to strengthen forecasting accuracy."
)

st.markdown('</div>', unsafe_allow_html=True)
