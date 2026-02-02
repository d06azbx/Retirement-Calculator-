import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Set page to wide
st.set_page_config(page_title="Retirement Planner", layout="wide")

def main():
    st.title("Retirement Planner")

    # --- SIDEBAR: CORE ASSUMPTIONS ---
    with st.sidebar:
        st.header("Core Assumptions")
        curr_age = st.number_input("Current Age", value=25)
        ret_age = st.number_input("Retirement Age", value=50)
        end_age = st.number_input("Plan Until Age", value=85)
        
        st.divider()
        init_savings = st.number_input("Current Savings (₹)", value=0)
        monthly_invest = st.number_input("Current Monthly Investment (₹)", value=10000)
        step_up_pct = st.number_input("Annual Step-up in Savings (%)", value=5.0) / 100
        
        st.divider()
        monthly_exp_today = st.number_input("Monthly Expense (Today's rate ₹)", value=50000)
        inflation_pct = st.number_input("Annual Inflation (%)", value=5.0) / 100

    # --- MAIN AREA: INVESTMENT & TAX APPROACH ---
    st.header("Investment & Tax Approach")
    
    # Asset definition
    assets = ["Fixed Returns", "Large Cap Mutual Funds", "Midcap Mutual Funds", "Smallcap Mutual funds"]
    def_returns = [0.07, 0.12, 0.15, 0.18]
    def_taxes = [0.30, 0.20, 0.20, 0.20]
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Earning Phase")
        e_shares = [0.20, 0.40, 0.30, 0.10]
        e_data = []
        h1, h2, h3, h4 = st.columns([2, 1, 1, 1])
        h1.caption("Asset Type")
        h2.caption("Return %")
        h3.caption("Tax %")
        h4.caption("Share %")
        
        for i, name in enumerate(assets):
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            c1.write(name)
            r = c2.number_input("Ret", value=int(def_returns[i]*100), key=f"er_{i}", label_visibility="collapsed") / 100
            t = c3.number_input("Tax", value=int(def_taxes[i]*100), key=f"et_{i}", label_visibility="collapsed") / 100
            s = c4.number_input("Shr", value=int(e_shares[i]*100), key=f"es_{i}", label_visibility="collapsed") / 100
            e_data.append({"r": r, "t": t, "s": s})
        
        w_ret_e = sum(d['r'] * d['s'] for d in e_data)
        w_tax_e = sum(d['t'] * d['s'] for d in e_data)
        st.info(f"**Weighted Return:** {w_ret_e:.2%} | **Weighted Tax:** {w_tax_e:.2%}")

    with col2:
        st.subheader("Retirement Phase")
        r_shares = [0.0, 1.0, 0.0, 0.0]
        r_data = []
        h1, h2, h3, h4 = st.columns([2, 1, 1, 1])
        h1.caption("Asset Type")
        h2.caption("Return %")
        h3.caption("Tax %")
        h4.caption("Share %")
        
        for i, name in enumerate(assets):
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            c1.write(name)
            r = c2.number_input("Ret", value=int(def_returns[i]*100), key=f"rr_{i}", label_visibility="collapsed") / 100
            t = c3.number_input("Tax", value=int(def_taxes[i]*100), key=f"rt_{i}", label_visibility="collapsed") / 100
            s = c4.number_input("Shr", value=int(r_shares[i]*100), key=f"rs_{i}", label_visibility="collapsed") / 100
            r_data.append({"r": r, "t": t, "s": s})
        
        w_ret_r = sum(d['r'] * d['s'] for d in r_data)
        w_tax_r = sum(d['t'] * d['s'] for d in r_data)
        st.info(f"**Weighted Return:** {w_ret_r:.2%} | **Weighted Tax:** {w_tax_r:.2%}")

    # --- AI MODEL: PREDICT ENDING BALANCE ---
    # Generate training data (Monte Carlo style)
    np.random.seed(42)
    train_X, train_y = [], []
    for _ in range(1000):
        start = np.random.uniform(0, 1_000_000)
        invest = np.random.uniform(0, 1_00_000)
        exp = np.random.uniform(0, 1_50_000)
        ret = np.random.uniform(0.05, 0.18)
        bal = start * (1 + ret) + invest - exp
        train_X.append([start, invest, exp, ret])
        train_y.append(bal)
    
    model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    model.fit(train_X, train_y)

    # --- CALCULATION USING AI ---
    results = []
    current_bal = init_savings
    annual_saving = monthly_invest * 12

    for age in range(curr_age, 101):
        if age < ret_age:
            status = "Earning"
            rate = w_ret_e
            inv = annual_saving if age == curr_age else annual_saving * ((1 + step_up_pct) ** (age - curr_age))
            exp = 0
        elif age < end_age:
            status = "Retired"
            rate = w_ret_r
            inv = 0
            exp = (monthly_exp_today * 12) * ((1 + inflation_pct) ** (age - curr_age))
        else:
            status = "Dead"
            rate, inv, exp, current_bal = 0, 0, 0, 0

        start_bal = current_bal
        if status != "Dead":
            # Predict ending balance using AI model
            end_bal = model.predict([[start_bal, inv, exp, rate]])[0]
        else:
            end_bal = 0
        
        results.append({
            "Age": age, "Status": status, "Starting Saving": start_bal,
            "Investment": inv, "Expenses": exp, "Ending Saving": end_bal
        })
        current_bal = end_bal

    df = pd.DataFrame(results)

    # --- DASHBOARD ---
    st.divider()
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.subheader("Summary")
        retirement_entry = df[df['Age'] == ret_age]
        if not retirement_entry.empty:
            corpus = retirement_entry['Starting Saving'].values[0]
            st.metric("Retirement Corpus", f"₹{corpus:,.0f}")
        
        fail_check = df[(df['Status'] == 'Retired') & (df['Ending Saving'] < 0)]
        if not fail_check.empty:
            st.error(f"⚠️ Funds exhausted at age {fail_check.iloc[0]['Age']}")
        else:
            st.success("✅ Plan is sustainable")
            
        fig_pie = go.Figure(data=[go.Pie(labels=assets, values=e_shares, hole=.3)])
        fig_pie.update_layout(title="Earning Asset Mix", height=300, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with res_col2:
        st.subheader("Wealth Projection")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Age'], y=df['Ending Saving'], fill='tozeroy', name="Net Wealth", line_color='#1f77b4'))
        fig.update_layout(height=450, margin=dict(l=0, r=0, t=20, b=0), yaxis_title="Savings (₹)")
        st.plotly_chart(fig, use_container_width=True)

    # --- FULL DATA TABLE ---
    with st.expander("View Detailed Annual Breakdown (Excel View)"):
        formatted_df = df.copy()
        for col in ["Starting Saving", "Investment", "Expenses", "Ending Saving"]:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"₹{x:,.0f}")
        st.table(formatted_df)

if __name__ == "__main__":
    main()
