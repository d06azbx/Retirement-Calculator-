import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Retirement Planner", layout="wide")
st.title("AI-Based Retirement Planner")

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
with st.sidebar:
    st.header("User Inputs")

    curr_age = st.number_input("Current Age", 18, 60, 25)
    ret_age = st.number_input("Retirement Age", 40, 70, 60)
    end_age = st.number_input("Life Expectancy", 70, 100, 85)

    monthly_invest = st.number_input("Monthly Investment (₹)", 1000, 500000, 15000)
    step_up = st.number_input("Annual Step-up (%)", 0.0, 20.0, 8.0) / 100

    monthly_expense = st.number_input("Monthly Expense at Retirement (₹)", 5000, 500000, 40000)
    inflation = st.number_input("Inflation (%)", 3.0, 10.0, 6.0) / 100

    risk_tolerance = st.slider("Risk Tolerance", 0.0, 1.0, 0.7)

# -------------------------------
# AI MODEL (SIMULATED TRAINING)
# -------------------------------
np.random.seed(42)

X = []
y = []

for age in range(20, 60):
    for risk in np.linspace(0.2, 1.0, 5):
        years_left = max(1, 60 - age)
        equity = min(0.9, max(0.2, risk * (years_left / 30)))
        X.append([age, years_left, risk])
        y.append(equity)

X = np.array(X)
y = np.array(y)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

years_to_ret = max(1, ret_age - curr_age)
equity_pct = model.predict([[curr_age, years_to_ret, risk_tolerance]])[0]
equity_pct = np.clip(equity_pct, 0.2, 0.8)
debt_pct = 1 - equity_pct

# -------------------------------
# RETURNS
# -------------------------------
equity_return = 0.12
debt_return = 0.07

portfolio_return = equity_pct * equity_return + debt_pct * debt_return

# -------------------------------
# SIMULATION
# -------------------------------
ages = []
corpus_values = []

corpus = 0
annual_invest = monthly_invest * 12
annual_expense = monthly_expense * 12

for age in range(curr_age, end_age + 1):

    # EARNING PHASE
    if age < ret_age:
        corpus += annual_invest
        annual_invest *= (1 + step_up)

    # RETIREMENT PHASE
    else:
        corpus -= annual_expense
        annual_expense *= (1 + inflation)

    # Apply portfolio returns
    if corpus > 0:
        corpus *= (1 + portfolio_return)

    corpus = max(corpus, 0)

    ages.append(age)
    corpus_values.append(corpus)

# -------------------------------
# SUSTAINABILITY CHECK
# -------------------------------
sustainable = corpus_values[-1] > 0

# -------------------------------
# RESULTS
# -------------------------------
st.subheader("Results")

col1, col2 = st.columns(2)

with col1:
    st.metric("Final Retirement Corpus", f"₹{corpus_values[-1]:,.0f}")
    st.success("Plan is sustainable ✅" if sustainable else "Plan is NOT sustainable ❌")

with col2:
    st.write("### AI Recommended Portfolio Mix")
    alloc_df = pd.DataFrame({
        "Asset": ["Equity", "Debt"],
        "Allocation %": [equity_pct * 100, debt_pct * 100]
    })
    st.dataframe(alloc_df, use_container_width=True)

# -------------------------------
# CHART
# -------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ages,
    y=corpus_values,
    fill='tozeroy',
    mode='lines',
    name="Corpus Growth"
))

fig.update_layout(
    title="Corpus Over Time",
    xaxis_title="Age",
    yaxis_title="Corpus (₹)",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)
