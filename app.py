import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Retirement Planner", layout="wide")

# -------------------------------------------------
# AI / ML HELPERS (NON-INTRUSIVE)
# -------------------------------------------------
def ai_insights(df, w_ret_e, w_ret_r, inflation_pct, ret_age):
    insights = []

    if w_ret_r < inflation_pct:
        insights.append("⚠️ Post-retirement returns are lower than inflation. Purchasing power erosion risk detected.")

    fail = df[(df['Status'] == 'Retired') & (df['Ending Saving'] < 0)]
    if not fail.empty:
        insights.append(f"❌ Retirement funds exhaust at age {int(fail.iloc[0]['Age'])}.")
    else:
        insights.append("✅ Retirement corpus sustains through the planned lifetime.")

    if w_ret_e < 0.10:
        insights.append("📉 Earning phase return is conservative. Long-term wealth accumulation may be suboptimal.")

    if ret_age < 55:
        insights.append("⏳ Early retirement increases longevity risk. Consider delaying retirement or increasing savings.")

    return insights


def monte_carlo_simulation(df, simulations=1000):
    final_balances = []

    for _ in range(simulations):
        bal = df.iloc[0]['Starting Saving']

        for _, row in df.iterrows():
            if row['Status'] == "Dead":
                break

            # Randomized return (ML-style stochastic behavior)
            random_return = np.random.normal(loc=0.11, scale=0.05)
            bal = bal * (1 + random_return) + row['Investment'] - row['Expenses']

        final_balances.append(bal)

    return final_balances


def retirement_score(df, ret_age):
    fail = df[(df['Status'] == 'Retired') & (df['Ending Saving'] < 0)]

    if not fail.empty:
        years_survived = fail.iloc[0]['Age'] - ret_age
        return max(25, 100 - (30 - years_survived) * 2)

    return 90


def ai_financial_coach(df, score):
    messages = []

    if score >= 85:
        messages.append("🟢 Excellent retirement readiness. Your current plan is well-structured.")
    elif score >= 65:
        messages.append("🟡 Moderate readiness. Minor tuning in savings or asset mix can improve outcomes.")
    else:
        messages.append("🔴 Low readiness. Action required: increase investments or delay retirement.")

    messages.append("📌 AI Recommendation: Diversify post-retirement assets to manage inflation and longevity risk.")
    messages.append("📌 AI Recommendation: Maintain at least 2–3 years of expenses in low-volatility instruments.")

    return messages


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

    # --------- RETIREMENT PHASE ---------
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

    # ---------------- CORE CALCULATION ENGINE (UNCHANGED) ----------------
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

    # ---------------- DASHBOARD ----------------
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
            st.error(f"⚠️ Funds exhausted at age {int(fail_check.iloc[0]['Age'])}")
        else:
            st.success("✅ Plan is sustainable")

        fig_pie = go.Figure(data=[go.Pie(labels=assets, values=[d['s'] for d in e_data], hole=.3)])
        fig_pie.update_layout(title="Earning Asset Mix", height=300, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with res_col2:
        st.subheader("Wealth Projection")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Age'],
            y=df['Ending Saving'],
            fill='tozeroy',
            name="Net Wealth"
        ))
        fig.update_layout(height=450, margin=dict(l=0, r=0, t=20, b=0), yaxis_title="Savings (₹)")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- AI / ML DASHBOARD ----------------
    st.divider()
    st.header("🤖 AI-Powered Analysis")

    ai_col1, ai_col2 = st.columns(2)

    with ai_col1:
        st.subheader("AI Insights")
        for msg in ai_insights(df, w_ret_e, w_ret_r, inflation_pct, ret_age):
            st.write(msg)

        score = retirement_score(df, ret_age)
        st.metric("Retirement Readiness Score", f"{score}/100")

    with ai_col2:
        st.subheader("Monte Carlo Risk Simulation")
        mc_results = monte_carlo_simulation(df)
        failure_prob = sum(b < 0 for b in mc_results) / len(mc_results)
        st.metric("Probability of Corpus Failure", f"{failure_prob:.1%}")

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(x=mc_results, nbinsx=40))
        fig_mc.update_layout(height=350, title="Distribution of Final Wealth")
        st.plotly_chart(fig_mc, use_container_width=True)

    st.subheader("🧠 AI Financial Coach")
    for advice in ai_financial_coach(df, score):
        st.write(advice)

    # ---------------- DATA TABLE ----------------
    with st.expander("View Detailed Annual Breakdown (Excel View)"):
        formatted_df = df.copy()
        for col in ["Starting Saving", "Investment", "Expenses", "Ending Saving"]:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"₹{x:,.0f}")
        st.table(formatted_df)


if __name__ == "__main__":
    main()
