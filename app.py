import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# Set page to wide
st.set_page_config(page_title="Retirement Planner", layout="wide")

def main():
    st.title("Retirement Planner")

    # ================= SIDEBAR =================
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

        # ===== ML UPLOAD =====
        st.divider()
        st.header("ML: Learn Returns from Data")
        uploaded_files = st.file_uploader(
            "Upload CSV / Excel files",
            type=["csv", "xlsx"],
            accept_multiple_files=True
        )

    # ================= ML LEARNING ENGINE =================
    learned_return = None

    if uploaded_files:
        returns = []

        for file in uploaded_files:
            if file.name.endswith(".csv"):
                df_ml = pd.read_csv(file)
            else:
                df_ml = pd.read_excel(file)

            # Case 1: Return column exists
            if "Return" in df_ml.columns:
                returns.extend(df_ml["Return"].dropna().values)

            # Case 2: Price data → compute returns
            else:
                price_col = None
                for col in df_ml.columns:
                    if col.lower() in ["close", "price", "nifty", "index"]:
                        price_col = col
                        break

                if price_col:
                    price_returns = df_ml[price_col].pct_change().dropna()
                    returns.extend(price_returns.values)

        if len(returns) > 10:
            X = np.arange(len(returns)).reshape(-1, 1)
            y = np.array(returns)

            model = LinearRegression()
            model.fit(X, y)

            learned_return = model.predict([[len(returns)]])[0]

            st.success(f"📊 ML Learned Expected Return: {learned_return:.2%}")
        else:
            st.warning("Not enough data for ML learning (need >10 rows)")

    # ================= INVESTMENT & TAX APPROACH =================
    st.header("Investment & Tax Approach")

    assets = [
        "Fixed Returns",
        "Large Cap Mutual Funds",
        "Midcap Mutual Funds",
        "Smallcap Mutual funds"
    ]

    def_returns = [0.07, 0.12, 0.15, 0.18]
    def_taxes = [0.30, 0.20, 0.20, 0.20]

    col1, col2 = st.columns(2)

    # -------- EARNING PHASE --------
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

            base_ret = learned_return if learned_return else def_returns[i]

            r = c2.number_input(
                "Ret", value=int(base_ret * 100),
                key=f"er_{i}", label_visibility="collapsed"
            ) / 100

            t = c3.number_input(
                "Tax", value=int(def_taxes[i] * 100),
                key=f"et_{i}", label_visibility="collapsed"
            ) / 100

            s = c4.number_input(
                "Shr", value=int(e_shares[i] * 100),
                key=f"es_{i}", label_visibility="collapsed"
            ) / 100

            e_data.append({"r": r, "t": t, "s": s})

        w_ret_e = sum(d["r"] * d["s"] for d in e_data)
        w_tax_e = sum(d["t"] * d["s"] for d in e_data)

        st.info(f"**Weighted Return:** {w_ret_e:.2%} | **Weighted Tax:** {w_tax_e:.2%}")

    # -------- RETIREMENT PHASE --------
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

            base_ret = learned_return if learned_return else def_returns[i]

            r = c2.number_input(
                "Ret", value=int(base_ret * 100),
                key=f"rr_{i}", label_visibility="collapsed"
            ) / 100

            t = c3.number_input(
                "Tax", value=int(def_taxes[i] * 100),
                key=f"rt_{i}", label_visibility="collapsed"
            ) / 100

            s = c4.number_input(
                "Shr", value=int(r_shares[i] * 100),
                key=f"rs_{i}", label_visibility="collapsed"
            ) / 100

            r_data.append({"r": r, "t": t, "s": s})

        w_ret_r = sum(d["r"] * d["s"] for d in r_data)
        w_tax_r = sum(d["t"] * d["s"] for d in r_data)

        st.info(f"**Weighted Return:** {w_ret_r:.2%} | **Weighted Tax:** {w_tax_r:.2%}")

    # ================= CALCULATION ENGINE (UNCHANGED) =================
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

    # ================= OUTPUT =================
    st.divider()
    colA, colB = st.columns([1, 2])

    with colA:
        st.subheader("Summary")
        corpus = df[df["Age"] == ret_age]["Starting Saving"].values[0]
        st.metric("Retirement Corpus", f"₹{corpus:,.0f}")

    with colB:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Age"],
            y=df["Ending Saving"],
            fill="tozeroy",
            name="Net Wealth"
        ))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Detailed Annual Breakdown"):
        show_df = df.copy()
        for col in ["Starting Saving", "Investment", "Expenses", "Ending Saving"]:
            show_df[col] = show_df[col].apply(lambda x: f"₹{x:,.0f}")
        st.table(show_df)

if __name__ == "__main__":
    main()
