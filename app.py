import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Retirement Planner", layout="wide")

# -------------------------------------------------
# ML DATA GENERATION
# -------------------------------------------------
def generate_ml_dataset(base_df, samples=1500):
    rows = []

    for _ in range(samples):
        ret = np.random.normal(0.11, 0.04)
        infl = np.random.normal(0.05, 0.01)
        sip = np.random.randint(8000, 20000)

        bal = base_df.iloc[0]['Starting Saving']
        exhausted_age = None

        for _, r in base_df.iterrows():
            bal = bal * (1 + ret) + sip * 12 - r['Expenses']
            if bal < 0 and exhausted_age is None:
                exhausted_age = r['Age']

        # Failure only if exhausted before 75
        failed = int(exhausted_age is not None and exhausted_age < 75)

        rows.append({
            "return": ret,
            "inflation": infl,
            "sip": sip,
            "final_balance": bal,
            "failed": failed
        })

    return pd.DataFrame(rows)


# -------------------------------------------------
# MAIN APP
# -------------------------------------------------
def main():
    st.title("Retirement Planner")

    # ---------------- SIDEBAR ----------------
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

    # ---------------- INVESTMENT SETUP ----------------
    st.header("Investment & Tax Approach")

    assets = ["Fixed Returns", "Large Cap Mutual Funds", "Midcap Mutual Funds", "Smallcap Mutual funds"]
    def_returns = [0.07, 0.12, 0.15, 0.18]
    def_taxes = [0.30, 0.20, 0.20, 0.20]

    col1, col2 = st.columns(2)

    # --------- EARNING PHASE ---------
    with col1:
        st.subheader("Earning Phase")
        e_shares = [0.20, 0.40, 0.30, 0.10]
        e_data = []

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

    # --------- RETIREMENT PHASE ---------
    with col2:
        st.subheader("Retirement Phase")
        r_shares = [0.0, 1.0, 0.0, 0.0]
        r_data = []

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

    # ---------------- CORE CALCULATION ENGINE ----------------
    results = []
    current_bal = init_savings
    annual_saving = monthly_invest * 12

    for age in range(curr_age, 101):
        if age < ret_age:
            status = "Earning"
            rate = w_ret_e
            inv = annual_saving * ((1 + step_up_pct) ** (age - curr_age))
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
        end_bal = start_bal * (1 + rate) + inv - exp if status != "Dead" else 0

        results.append({
            "Age": age,
            "Status": status,
            "Starting Saving": start_bal,
            "Investment": inv,
            "Expenses": exp,
            "Ending Saving": end_bal
        })

        current_bal = end_bal

    df = pd.DataFrame(results)

    # ---------------- SUMMARY ----------------
    st.divider()
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.subheader("Summary")

        retirement_entry = df[df['Age'] == ret_age]
        if not retirement_entry.empty:
            corpus = retirement_entry['Starting Saving'].values[0]
            st.metric("Retirement Corpus", f"₹{corpus:,.0f}")

        fail_check = df[(df['Status'] == 'Retired') & (df['Ending Saving'] < 0) & (df['Age'] < 75)]
        if not fail_check.empty:
            st.error(f"⚠️ Funds exhausted at age {int(fail_check.iloc[0]['Age'])}")
        else:
            st.success("✅ Plan is acceptable (funds last till at least 75)")

    with res_col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Age'],
            y=df['Ending Saving'],
            fill='tozeroy',
            name="Net Wealth"
        ))
        fig.update_layout(height=450, yaxis_title="Savings (₹)")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- ML SECTION ----------------
    st.divider()
    st.header("🤖 Machine Learning Risk Analysis")

    ml_df = generate_ml_dataset(df)

    X = ml_df[['return', 'inflation', 'sip']]
    y_fail = ml_df['failed']
    y_balance = ml_df['final_balance']

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X, y_balance)
    lr_pred = lr.predict([[w_ret_e, inflation_pct, monthly_invest]])[0]

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    rf.fit(X, y_fail)
    rf_prob = rf.predict_proba([[w_ret_e, inflation_pct, monthly_invest]])[0][1]

    col_ml1, col_ml2 = st.columns(2)

    with col_ml1:
        st.metric("Predicted Final Corpus (Linear Regression)", f"₹{lr_pred:,.0f}")

    with col_ml2:
        st.metric("Failure Probability Before Age 75 (Random Forest)", f"{rf_prob:.1%}")

    # ---------------- DATA TABLE ----------------
    with st.expander("View Detailed Annual Breakdown (Excel View)"):
        formatted_df = df.copy()
        for col in ["Starting Saving", "Investment", "Expenses", "Ending Saving"]:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"₹{x:,.0f}")
        st.table(formatted_df)


if __name__ == "__main__":
    main()
