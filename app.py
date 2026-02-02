import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Page setup
st.set_page_config(page_title="AI Retirement Planner", layout="wide")

def main():
    st.title("AI-Powered Retirement Planner")

    # ---------------- SIDEBAR: CORE ASSUMPTIONS ----------------
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
        monthly_exp_today = st.number_input("Monthly Expense (Today's ₹)", value=50000)
        inflation_pct = st.number_input("Annual Inflation (%)", value=5.0) / 100

    # ---------------- ASSET DEFINITIONS ----------------
    assets = [
        "Fixed Returns",
        "Large Cap Mutual Funds",
        "Midcap Mutual Funds",
        "Smallcap Mutual funds"
    ]

    def_returns = [0.07, 0.12, 0.15, 0.18]
    def_taxes = [0.30, 0.20, 0.20, 0.20]

    st.header("Investment & Tax Approach")
    col1, col2 = st.columns(2)

    # ---------------- EARNING PHASE ----------------
    with col1:
        st.subheader("Earning Phase Allocation")

        e_shares = [0.20, 0.40, 0.30, 0.10]
        e_data = []

        h1, h2, h3, h4 = st.columns([2, 1, 1, 1])
        h1.caption("Asset")
        h2.caption("Return %")
        h3.caption("Tax %")
        h4.caption("Share %")

        for i, asset in enumerate(assets):
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            c1.write(asset)
            r = c2.number_input("R", value=int(def_returns[i]*100), key=f"er_{i}", label_visibility="collapsed") / 100
            t = c3.number_input("T", value=int(def_taxes[i]*100), key=f"et_{i}", label_visibility="collapsed") / 100
            s = c4.number_input("S", value=int(e_shares[i]*100), key=f"es_{i}", label_visibility="collapsed") / 100
            e_data.append({"r": r, "t": t, "s": s})

        w_ret_e = sum(d["r"] * d["s"] for d in e_data)
        w_tax_e = sum(d["t"] * d["s"] for d in e_data)

        st.info(f"Weighted Return: {w_ret_e:.2%} | Weighted Tax: {w_tax_e:.2%}")

    # ---------------- RETIREMENT PHASE ----------------
    with col2:
        st.subheader("Retirement Phase Allocation")

        r_shares = [0.0, 1.0, 0.0, 0.0]
        r_data = []

        h1, h2, h3, h4 = st.columns([2, 1, 1, 1])
        h1.caption("Asset")
        h2.caption("Return %")
        h3.caption("Tax %")
        h4.caption("Share %")

        for i, asset in enumerate(assets):
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            c1.write(asset)
            r = c2.number_input("R", value=int(def_returns[i]*100), key=f"rr_{i}", label_visibility="collapsed") / 100
            t = c3.number_input("T", value=int(def_taxes[i]*100), key=f"rt_{i}", label_visibility="collapsed") / 100
            s = c4.number_input("S", value=int(r_shares[i]*100), key=f"rs_{i}", label_visibility="collapsed") / 100
            r_data.append({"r": r, "t": t, "s": s})

        w_ret_r = sum(d["r"] * d["s"] for d in r_data)
        w_tax_r = sum(d["t"] * d["s"] for d in r_data)

        st.info(f"Weighted Return: {w_ret_r:.2%} | Weighted Tax: {w_tax_r:.2%}")

    # ---------------- AI PORTFOLIO RECOMMENDATION ----------------
    def ai_portfolio_recommendation(curr_age, ret_age):
        years_left = ret_age - curr_age
        risk_score = min(max(years_left / 30, 0), 1)

        fixed = 0.6 - 0.4 * risk_score
        large = 0.25 + 0.2 * risk_score
        mid = 0.10 + 0.15 * risk_score
        small = 0.05 + 0.05 * risk_score

        alloc = np.array([fixed, large, mid, small])
        alloc = alloc / alloc.sum()

        return alloc, risk_score

    alloc, risk_score = ai_portfolio_recommendation(curr_age, ret_age)

    st.subheader("🤖 AI Recommended Portfolio Mix")

    ai_df = pd.DataFrame({
        "Asset Class": assets,
        "Recommended Allocation %": alloc * 100
    })

    st.table(ai_df.style.format({"Recommended Allocation %": "{:.1f}%"}))
    st.info(f"AI Risk Score: {risk_score:.2f} (Higher = higher equity exposure)")

    # ---------------- CALCULATION ENGINE ----------------
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
        st.subheader("Summary")
        corpus = df[df["Age"] == ret_age]["Starting Saving"].values[0]
        st.metric("Retirement Corpus", f"₹{corpus:,.0f}")

        fail = df[(df["Status"] == "Retired") & (df["Ending Saving"] < 0)]
        if not fail.empty:
            st.error(f"Funds exhausted at age {fail.iloc[0]['Age']}")
        else:
            st.success("Plan is sustainable")

        pie = go.Figure(go.Pie(labels=assets, values=alloc, hole=0.4))
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
        fig.update_layout(height=450, yaxis_title="₹ Savings")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- TABLE ----------------
    with st.expander("Detailed Year-wise Table"):
        fdf = df.copy()
        for c in ["Starting Saving", "Investment", "Expenses", "Ending Saving"]:
            fdf[c] = fdf[c].apply(lambda x: f"₹{x:,.0f}")
        st.table(fdf)

if __name__ == "__main__":
    main()
