# SME Mental Health & Economic Impact — MVP (Streamlit)

This prototype demonstrates an AI-enabled **Decision Support Agent** for workforce mental health in Small and Medium-sized Enterprises (SMEs).

The app ingests SME-level indicators (1–10 scores for absenteeism, burnout, stress, plus employees and wages), estimates **lost workdays** and **economic loss (CAD)**, and simulates the **impact of interventions** (e.g., reducing burnout scores) across Canadian provinces.

## What funders can see in the demo

- **Synthetic SME dataset** by province and sector (no real personal data).
- **Per-SME economic loss** estimates:
  - Lost days per employee per month (absenteeism + burnout-driven presenteeism).
  - Estimated monthly loss per SME in CAD.
- **Province-level aggregation**:
  - Total employees covered.
  - Average mental-health scores.
  - Total estimated monthly economic loss by province.
- **Scenario simulation**:
  - Test simple policy levers (reduce stress/burnout/absenteeism scores by N points).
  - See updated economic loss and **estimated CAD savings** across provinces.
- **Downloadable reports**:
  - CSV exports for province-level and SME-level simulated results, suitable for sharing with policymakers and analysts.

## How this would scale beyond the MVP

With real SME data and open labour statistics (e.g., Statistics Canada), this tool can:

- Quantify the **economic burden of mental-health-related absenteeism and burnout**.
- Help governments and SME associations **prioritize interventions** where they have the greatest economic and social return.
- Provide a transparent, configurable framework for **ongoing monitoring**, without exposing individual-level identities.

This MVP is designed as a starting point for a **pilot with 10–30 SMEs in one province**, leading to a full-scale platform integrating secure data pipelines, privacy safeguards, and richer forecasting models.
