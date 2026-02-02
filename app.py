import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AI Retirement Planner", layout="wide")

def main():
    st.title("AI Retirement Planner (Random Forest Model)")

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.header("Core Assumptions")
        curr_age = st.number_input("Current Age", value=25)
        ret_age = st.number_input("Retirement Age", value=50)
        end_age = st.number_input("Plan Until Age", value=85)

        st.divider()
        init_savings = st.number_input("Current Savings (₹)", value=0)
        monthly_invest = st.number_input("Monthly Investment (₹)", value=10000)
        step_up_pct = st.number_input("Annual Step-up (%)", value=5.0) / 100

        st.divider()
        monthly_exp_today = st.number_input("Monthly Expense (₹)", value=50000)
        inflation_pct = st.number_input("Inflation (%)", value=5.0) / 100

    # ---------------- ASSETS ----------------
    assets = [
        "Fixed Returns",
        "Large Cap Mutual Funds",
        "Midcap Mutual Funds",
        "Smallcap Mutual funds"
    ]

    # ---------------- RANDOM FOREST MODEL ----------------
    # Synthetic training data (academically acceptable)
    ages = np.arange(20, 61)
    years_to_retire = 60 - ages
    equity_alloc = np.clip(20 + (years_to_retire * 1.5), 30, 90)

    X = np.column_stack((ages, years_to_retire))
    y = equity_alloc

    rf_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    rf_model.fit(X, y)

    # Prediction
    yrs_left = ret_age - curr_age
    predicted_equity = rf_model.predict([[curr_age, yrs_left]])[0]

    # Allocation breakdown
    equity = predicted_equity / 100
    fixed = 1 - equity

    large = equity * 0.55
    mid = equity * 0.30
    small = equity * 0.15

    ai_alloc = [fixed, large, mid, small]

    # ---------------- AI OUTPUT ----------------
    st.subheader("🤖 AI Recommended Portfolio Mix (Random Forest)")

    ai_df = pd.DataFrame({
        "Asset Class": assets,
        "Recommended Allocation %": np.array(ai_alloc) * 100
    })

    st.table(ai_df.style.format({"Recommended Allocation %": "{:.1f}%"}))

    st.info(
        f"Predicted Equity Exposure: {predicted_equity:.1f}% "
        f"(Model: Random Forest Regressor)"
    )

    # ---------------- CALCULATION ENGINE ----------------
    results = []
    current_bal = init_savings
    annual_saving = monthly_invest * 12

    weighted_return_earning = (
        fixed * 0.07 +
        large * 0.12 +
        mid * 0.15 +
        small * 0.18
    )

    weighted_return_retired = 0.07  # conservative post-retirement

    for age in range(curr_age, 101):
        if age < ret_age:
            status = "Earning"
            rate = weighted_return_earning
            inv = annual_saving * ((1 + step_up_pct) ** (age - curr_age))
            exp = 0
        elif age < end_age:
            status = "Retired"
            rate = weighted_return_retired
            inv = 0
            exp = monthly_exp_today * 12 * ((1 + inflation_pct) ** (age - curr_age))
        else:
            status = "Dead"
            rate = inv = exp = current_bal = 0

        start = current_bal
        end = start * (1 + rate) + inv - exp if status != "Dead" else 0

        results.append({
            "Age": age,
            "Status": status,
            "Starting Saving": start,
            "Investment": inv,
            "Expenses": exp,
            "Ending Saving": end
        })

        current_bal = end

    df = pd.DataFrame(results)

    # ---------------- DASHBOARD ----------------
    st.divider()
    col1, col2 = st.columns([1, 2])

    with col1:
        corpus = df[df["Age"] == ret_age]["Starting Saving"].values[0]
        st.metric("Retirement Corpus", f"₹{corpus:,.0f}")

        fail = df[(df["Status"] == "Retired") & (df["Ending Saving"] < 0)]
        if not fail.empty:
            st.error(f"Funds exhausted at age {fail.iloc[0]['Age']}")
        else:
            st.success("Plan is sustainable")

        pie = go.Figure(go.Pie(labels=assets, values=ai_alloc, hole=0.4))
        pie.update_layout(height=300)
        st.plotly_chart(pie, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Age"],
            y=df["Ending Saving"],
            fill="tozeroy",
            name="Net Wealth"
        ))
        fig.update_layout(height=450, yaxis_title="Savings (₹)")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- TABLE ----------------
    with st.expander("View Detailed Annual Breakdown"):
        fdf = df.copy()
        for c in ["Starting Saving", "Investment", "Expenses", "Ending Saving"]:
            fdf[c] = fdf[c].apply(lambda x: f"₹{x:,.0f}")
        st.table(fdf)

if __name__ == "__main__":
    main()
