# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Portfolio Analyzer with Married Put", layout="wide")
st.title("📈 Monte Carlo Portfolio Optimizer with Married Put Insurance")

# Upload section
st.sidebar.header("📂 Upload Asset CSV Files")
uploaded_files = st.sidebar.file_uploader("Upload CSV files (each with 'Date' and 'Price')", type=['csv'], accept_multiple_files=True)

# Period selection
period = st.sidebar.selectbox("Return Analysis Period", ['Monthly', 'Quarterly', 'Semi-Annually'])
resample_rule = {'Monthly': 'M', 'Quarterly': 'Q', 'Semi-Annually': '2Q'}[period]
annual_factor = {'Monthly': 12, 'Quarterly': 4, 'Semi-Annually': 2}[period]

if uploaded_files:
    prices_df = pd.DataFrame()
    asset_names = []
    insured_assets = {}

    for file in uploaded_files:
        name = file.name.split('.')[0]
        df = pd.read_csv(file)
        if 'Date' not in df.columns or 'Price' not in df.columns:
            st.warning(f"File {name} must contain 'Date' and 'Price' columns.")
            continue

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Price'] = df['Price'].astype(str).str.replace(',', '')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price'])
        df = df[['Date', 'Price']].set_index('Date')
        df.columns = [name]

        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')
        asset_names.append(name)

        with st.sidebar.expander(f"⚙️ Insurance Settings for {name}"):
            insured = st.checkbox(f"Enable insurance for {name}", key=f"insured_{name}")
            if insured:
                strike = st.number_input("Strike Price", value=100.0, step=1.0, key=f"strike_{name}")
                premium = st.number_input("Premium", value=5.0, step=0.1, key=f"premium_{name}")
                spot = st.number_input("Current Price", value=100.0, step=1.0, key=f"spot_{name}")
                amount = st.number_input("Asset Amount", value=1.0, step=1.0, key=f"amount_{name}")
                insured_assets[name] = {
                    'strike': strike,
                    'premium': premium,
                    'spot': spot,
                    'amount': amount
                }

    if prices_df.empty:
        st.error("❌ No valid data found for analysis.")
        st.stop()

    st.subheader("📊 Price Preview")
    st.dataframe(prices_df.tail())

    returns = prices_df.resample(resample_rule).last().pct_change().dropna()
    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor

    # Adjust risk for insured assets
    for asset in insured_assets:
        cov_matrix.loc[asset, asset] *= 0.25  # reduce risk if insured

    # Monte Carlo Simulation
    np.random.seed(42)
    n_portfolios = 5000
    n_assets = len(asset_names)
    results = np.zeros((3 + n_assets, n_portfolios))

    for i in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = port_return / port_std
        results[0, i] = port_return
        results[1, i] = port_std
        results[2, i] = sharpe
        results[3:, i] = weights

    # Find optimal portfolio near target risk
    target_risk = st.sidebar.slider("🎯 Target Portfolio Risk (%)", 5, 50, 25) / 100
    best_idx = np.argmin(np.abs(results[1] - target_risk))
    best_return = results[0, best_idx]
    best_risk = results[1, best_idx]
    best_sharpe = results[2, best_idx]
    best_weights = results[3:, best_idx]

    st.subheader("📈 Optimal Portfolio")
    st.write(f"**Annual Return:** {best_return:.2%}")
    st.write(f"**Annual Risk:** {best_risk:.2%}")
    st.write(f"**Sharpe Ratio:** {best_sharpe:.2f}")

    for i, name in enumerate(asset_names):
        st.write(f"🔹 {name}: {best_weights[i]*100:.2f}%")

    fig = px.scatter(
        x=results[1]*100, y=results[0]*100, color=results[2],
        labels={'x': 'Risk (%)', 'y': 'Return (%)'},
        title='Simulated Portfolios', color_continuous_scale='Viridis')
    fig.add_scatter(x=[best_risk*100], y=[best_return*100], mode='markers', marker=dict(size=12, color='red', symbol='star'), name='Optimal')
    st.plotly_chart(fig)

    # Plot Married Put
    for asset, data in insured_assets.items():
        st.subheader(f"💹 Married Put Strategy - {asset}")
        x = np.linspace(data['spot'] * 0.7, data['spot'] * 1.3, 300)
        asset_pnl = (x - data['spot']) * data['amount']
        put_pnl = np.where(x < data['strike'], data['strike'] - x, 0) * data['amount'] - data['premium'] * data['amount']
        total_pnl = asset_pnl + put_pnl

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=total_pnl, name="Married Put", line=dict(color='blue', width=2)))
        fig2.add_trace(go.Scatter(x=x, y=asset_pnl, name="Stock Only", line=dict(dash='dot')))
        fig2.add_trace(go.Scatter(x=x, y=put_pnl, name="Put Option", line=dict(dash='dot')))

        fig2.add_shape(type="rect", x0=x.min(), x1=data['spot'], y0=min(total_pnl.min(), 0), y1=0,
                      fillcolor="red", opacity=0.2, line_width=0)
        fig2.add_shape(type="rect", x0=data['spot'], x1=x.max(), y0=0, y1=max(total_pnl.max(), 0),
                      fillcolor="green", opacity=0.2, line_width=0)

        fig2.add_vline(x=data['strike'], line=dict(dash='dot', color='black'))
        fig2.add_vline(x=data['spot'], line=dict(dash='dot', color='green'))

        fig2.update_layout(
            title="Profit / Loss of Married Put",
            xaxis_title="Underlying Price at Expiration",
            yaxis_title="Profit / Loss",
            plot_bgcolor="white",
            hovermode="x unified"
        )

        st.plotly_chart(fig2)
else:
    st.warning("Please upload at least one CSV file.")
