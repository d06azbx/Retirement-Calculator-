import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Set page to wide
st.set_page_config(page_title="Professional ML Retirement Planner", layout="wide")

def main():
    st.title("🎯 Professional ML Retirement Planner")
    st.markdown("### Using Polynomial Regression (Degree 2) to model exponential wealth growth")

    # --- SIDEBAR: CORE ASSUMPTIONS ---
    with st.sidebar:
        st.header("🔑 Core Assumptions")
        curr_age = st.number_input("Current Age", value=25)
        ret_age = st.number_input("Retirement Age", value=50)
        end_age = st.number_input("Plan Until Age", value=85)
        
        st.divider()
        init_savings = st.number_input("Current Savings (₹)", value=0)
        monthly_invest = st.number_input("Current Monthly Investment (₹)", value=10000)
        step_up_pct = st.number_input("Annual Step-up (%)", value=5.0) / 100
        
        st.divider()
        monthly_exp_today = st.number_input("Monthly Expense (Today's ₹)", value=50000)
        inflation_pct = st.number_input("Annual Inflation (%)", value=5.0) / 100

    # --- INVESTMENT APPROACH ---
    st.header("📈 Investment Strategy")
    assets = ["Fixed Returns", "Large Cap Funds", "Midcap Funds", "Smallcap Funds"]
    def_returns = [0.07, 0.12, 0.15, 0.18]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Earning Phase")
        e_shares = [0.20, 0.40, 0.30, 0.10]
        e_rets = []
        for i, name in enumerate(assets):
            c1, c2 = st.columns([2,1])
            s = c2.number_input(f"Share", value=e_shares[i], key=f"es_{i}", label_visibility="collapsed")
            c1.write(f"{name} ({def_returns[i]*100}%)")
            e_rets.append(def_returns[i] * s)
        w_ret_e = sum(e_rets)

    with col2:
        st.subheader("Retirement Phase")
        r_shares = [0.0, 1.0, 0.0, 0.0]
        r_rets = []
        for i, name in enumerate(assets):
            c1, c2 = st.columns([2,1])
            s = c2.number_input(f"Share", value=r_shares[i], key=f"rs_{i}", label_visibility="collapsed")
            c1.write(f"{name} ({def_returns[i]*100}%)")
            r_rets.append(def_returns[i] * s)
        w_ret_r = sum(r_rets)

    # --- ML ENGINE (POLYNOMIAL REGRESSION) ---
    # 1. Create Synthetic Training Data (Time vs Growth)
    X_train = np.arange(0, 100).reshape(-1, 1) 
    y_train = (1 + 0.10)**X_train  # Base growth curve

    # 2. Transform to Polynomial Features (Degree 2 captures the curve)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X_train)

    # 3. Train the Model
    model = LinearRegression()
    model.fit(X_poly, y_train)

    # --- CALCULATION ENGINE ---
    results = []
    current_bal = init_savings
    annual_saving = monthly_invest * 12

    for age in range(curr_age, end_age + 1):
        year_idx = age - curr_age
        status = "Earning" if age < ret_age else "Retired"
        rate = w_ret_e if status == "Earning" else w_ret_r
        
        # ML PREDICTION: Instead of balance * (1+rate), we use the ML curve
        # We predict the growth multiplier for "1 year" at the specific rate
        X_test_poly = poly.transform(np.array([[1]]))
        ml_multiplier = model.predict(X_test_poly)[0][0]
        
        # Adjust ML multiplier to the specific user-selected weighted rate
        # (This ensures the ML model drives the growth while respecting user inputs)
        adjusted_ml_rate = 1 + (rate * (ml_multiplier - 1) / 0.10)

        inv = annual_saving * ((1 + step_up_pct) ** year_idx) if status == "Earning" else 0
        exp = (monthly_exp_today * 12) * ((1 + inflation_pct) ** year_idx) if status == "Retired" else 0
        
        start_bal = current_bal
        end_bal = (start_bal * adjusted_ml_rate) + inv - exp
        
        results.append({
            "Age": age, "Status": status, "Starting": start_bal,
            "Investment": inv, "Expenses": exp, "Ending": max(0, end_bal)
        })
        current_bal = end_bal

    df = pd.DataFrame(results)

    # --- DASHBOARD ---
    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Retirement Corpus", f"₹{df[df['Age']==ret_age]['Starting'].values[0]:,.0f}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Age'], y=df['Ending'], fill='tozeroy', name="Wealth Path (ML Predicted)"))
    fig.update_layout(title="Wealth Projection Curve", yaxis_title="Savings (₹)")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Data Table"):
        st.dataframe(df.style.format("{:,.0f}", subset=["Starting", "Investment", "Expenses", "Ending"]))

if __name__ == "__main__":
    main()
