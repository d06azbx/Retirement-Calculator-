import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# Page Configuration
st.set_page_config(page_title="Advanced ML Retirement Planner", layout="wide")

def main():
    st.title("🤖 Random Forest Retirement Planner")
    st.markdown("This model uses an **Ensemble Machine Learning** approach to forecast wealth.")

    # --- SIDEBAR: INPUTS ---
    with st.sidebar:
        st.header("🔑 Core Assumptions")
        curr_age = st.number_input("Current Age", value=25)
        ret_age = st.number_input("Retirement Age", value=50)
        end_age = st.number_input("Plan Until Age", value=85)
        st.divider()
        init_savings = st.number_input("Current Savings (₹)", value=0)
        monthly_invest = st.number_input("Monthly Investment (₹)", value=10000)
        step_up_pct = st.number_input("Annual Step-up (%)", value=5.0) / 100
        st.divider()
        monthly_exp_today = st.number_input("Monthly Expense (Today's ₹)", value=50000)
        inflation_pct = st.number_input("Annual Inflation (%)", value=5.0) / 100

    # --- PORTFOLIO RETURNS ---
    assets = ["Fixed Returns", "Large Cap", "Midcap", "Smallcap"]
    def_returns = [0.07, 0.12, 0.15, 0.18]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Earning Phase")
        e_shares = [0.20, 0.40, 0.30, 0.10]
        e_rets = []
        for i, name in enumerate(assets):
            r = st.number_input(f"{name} Ret (E)", value=def_returns[i], key=f"er_{i}")
            s = st.number_input(f"{name} Share (E)", value=e_shares[i], key=f"es_{i}")
            e_rets.append(r * s)
        w_ret_e = sum(e_rets)

    with col2:
        st.subheader("Retirement Phase")
        r_shares = [0.0, 1.0, 0.0, 0.0]
        r_rets = []
        for i, name in enumerate(assets):
            r = st.number_input(f"{name} Ret (R)", value=def_returns[i], key=f"rr_{i}")
            s = st.number_input(f"{name} Share (R)", value=r_shares[i], key=f"rs_{i}")
            r_rets.append(r * s)
        w_ret_r = sum(r_rets)

    # --- ML ENGINE (RANDOM FOREST) ---
    st.divider()
    # Create a training set of 1000 scenarios
    # Features: [AgeIndex, Rate, IsRetired]
    n_samples = 1000
    ages_train = np.random.randint(0, 60, n_samples)
    rates_train = np.random.uniform(0.05, 0.20, n_samples)
    retired_train = (ages_train > 25).astype(int)
    
    # Target: Growth Multiplier (Simulated math)
    y_train = (1 + rates_train) ** ages_train
    
    X_train = pd.DataFrame({'Age': ages_train, 'Rate': rates_train, 'Retired': retired_train})
    
    # Fit Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    st.success("✅ Random Forest Model Trained and Active")

    # --- CALCULATIONS ---
    results = []
    current_bal = init_savings
    annual_saving = monthly_invest * 12

    for age in range(curr_age, end_age + 1):
        year_idx = age - curr_age
        is_retired = 1 if age >= ret_age else 0
        rate = w_ret_r if is_retired else w_ret_e
        
        # ML Inference
        pred_input = pd.DataFrame({'Age': [1], 'Rate': [rate], 'Retired': [is_retired]})
        growth_multiplier = rf_model.predict(pred_input)[0]
        
        # Cash Flows
        inv = annual_saving * ((1 + step_up_pct) ** year_idx) if not is_retired else 0
        exp = (monthly_exp_today * 12) * ((1 + inflation_pct) ** year_idx) if is_retired else 0
        
        # Wealth formula using ML Growth
        start_bal = current_bal
        end_bal = (start_bal * growth_multiplier) + inv - exp
        
        results.append({"Age": age, "Starting": start_bal, "Ending": max(0, end_bal)})
        current_bal = end_bal

    # --- PLOT ---
    df = pd.DataFrame(results)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Age'], y=df['Ending'], mode='lines', fill='tozeroy', name="ML Forecast"))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df)

if __name__ == "__main__":
    main()
