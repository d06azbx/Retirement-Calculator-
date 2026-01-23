import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Page config
st.set_page_config(page_title="AI Retirement Planner", layout="wide")

def main():
    st.title("AI-Powered Retirement Planner")

    # ================= SIDEBAR =================
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
        monthly_exp_today = st.number_input("Monthly Expense Today (₹)", value=50000)
        inflation_pct = st.number_input("Inflation (%)", value=5.0) / 100

        st.divider()
        st.header("AI / Simulation Settings")
        enable_mc = st.checkbox("Enable Monte Carlo Simulation", True)
        mc_runs = st.number_input("Simulation Runs", 500, 5000, 1000, step=500)
        volatility = st.number_input("Return Volatility (%)", value=12.0) / 100

    # ================= ASSET CONFIG =================
    st.header("Investment Strategy")

    assets = ["Fixed Returns", "Large Cap MF", "Midcap MF", "Smallcap MF"]
    def_returns = [0.07, 0.12, 0.15, 0.18]
    def_taxes = [0.30, 0.20, 0.20, 0.20]

    col1, col2 = st.columns(2)

    # -------- EARNING PHASE --------
    with col1:
        st.subheader("Earning Phase Allocation")

        e_shares = [0.20, 0.40, 0.30, 0.10]
        e_data = []

        for i, name in enumerate(assets):
            c1, c2, c3 = st.columns(3)
            c1.write(name)
            r = c2.number_input("Return %", value=int(def_returns[i]*100), key=f"er{i}") / 100
            s = c3.number_input("Share %", value=int(e_shares[i]*100), key=f"es{i}") / 100
            e_data.append({"r": r, "s": s})

        w_ret_e = sum(d["r"] * d["s"] for d in e_data)
        st.info(f"Weighted Return (Earning): {w_ret_e:.2%}")

    # -------- RETIREMENT PHASE --------
    with col2:
        st.subheader("Retirement Phase Allocation")

        r_shares = [1.0, 0.0, 0.0, 0.0]
        r_data = []

        for i, name in enumerate(assets):
            c1, c2, c3 = st.columns(3)
            c1.write(name)
            r = c2.number_input("Return %", value=int(def_returns[i]*100), key=f"rr{i}") / 100
            s = c3.number_input("Share %", value=int(r_shares[i]*100), key=f"rs{i}") / 100
            r_data.append({"r": r, "s": s})

        w_ret_r = sum(d["r"] * d["s"] for d in r_data)
        st.info(f"Weighted Return (Retirement): {w_ret_r:.2%}")

    # ================= CORE CALCULATION =================
    results = []
    current_bal = init_savings
    annual_saving = monthly_invest * 12

    for age in range(curr_age, end_age + 1):
        if age < ret_age:
            status = "Earning"
            rate = w_ret_e
            inv = annual_saving * ((1 + step_up_pct) ** (age - curr_age))
            exp = 0
        else:
            status = "Retired"
            rate = w_ret_r
            inv = 0
            exp = (monthly_exp_today * 12) * ((1 + inflation_pct) ** (age - curr_age))

        start_bal = current_bal
        end_bal = start_bal * (1 + rate) + inv - exp

        results.append({
            "Age": age,
            "Status": status,
            "Starting Balance": start_bal,
            "Investment": inv,
            "Expenses": exp,
            "Ending Balance": end_bal
        })

        current_bal = end_bal

    df = pd.DataFrame(results)

    # ================= MONTE CARLO AI MODEL =================
    def monte_carlo_sim():
        final_balances = []
        ruin = 0

        for _ in range(mc_runs):
            bal = init_savings

            for age in range(curr_age, end_age):
                if age < ret_age:
                    mean_ret = w_ret_e
                    inv = annual_saving * ((1 + step_up_pct) ** (age - curr_age))
                    exp = 0
                else:
                    mean_ret = w_ret_r
                    inv = 0
                    exp = (monthly_exp_today * 12) * ((1 + inflation_pct) ** (age - curr_age))

                actual_return = np.random.normal(mean_ret, volatility)
                bal = bal * (1 + actual_return) + inv - exp

                if bal < 0:
                    ruin += 1
                    break

            final_balances.append(max(bal, 0))

        success_rate = (mc_runs - ruin) / mc_runs
        return final_balances, success_rate

    # ================= RESULTS =================
    st.divider()
    colA, colB = st.columns([1, 2])

    with colA:
        st.subheader("Summary")

        retirement_corpus = df[df["Age"] == ret_age]["Starting Balance"].values[0]
        st.metric("Corpus at Retirement", f"₹{retirement_corpus:,.0f}")

        if enable_mc:
            mc_results, success = monte_carlo_sim()
            st.metric("Retirement Success Probability", f"{success:.1%}")

    with colB:
        st.subheader("Wealth Projection")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Age"],
            y=df["Ending Balance"],
            fill="tozeroy",
            name="Deterministic Wealth"
        ))
        st.plotly_chart(fig, use_container_width=True)

        if enable_mc:
            fig2 = go.Figure()
            fig2.add_histogram(x=mc_results, nbinsx=40)
            fig2.update_layout(
                title="Monte Carlo: Final Wealth Distribution",
                xaxis_title="Final Wealth (₹)",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ================= DATA TABLE =================
    with st.expander("Detailed Year-wise Table"):
        display_df = df.copy()
        for col in ["Starting Balance", "Investment", "Expenses", "Ending Balance"]:
            display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(display_df)

if __name__ == "__main__":
    main()
