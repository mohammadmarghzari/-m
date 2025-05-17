import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

st.set_page_config(layout="wide")
st.title("ابزار تحلیل پرتفو و بیمه با آپشن پوت")

# --------------------- بخش بارگذاری داده ---------------------
st.sidebar.header("بارگذاری فایل دارایی‌ها (Date, Price)")
uploaded_files = st.sidebar.file_uploader("انتخاب فایل‌ها", accept_multiple_files=True, type=["csv"])

price_data = {}
for file in uploaded_files:
    df = pd.read_csv(file)
    df.columns = [col.strip().lower() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['price'] = df['price'].astype(str).str.replace(',', '').astype(float)
    name = file.name.split(".")[0]
    price_data[name] = df.set_index('date')['price']

if price_data:
    prices_df = pd.concat(price_data.values(), axis=1)
    prices_df.columns = price_data.keys()
    returns = prices_df.pct_change().dropna()

    st.subheader("تنظیمات بیمه (Protective Put)")
    insurance_settings = {}
    for asset in prices_df.columns:
        with st.expander(f"📉 {asset}"):
            insured = st.checkbox(f"آیا بیمه دارد؟", key=f"{asset}_insured")
            if insured:
                price = st.number_input(f"قیمت خرید دارایی پایه", key=f"{asset}_base", value=prices_df[asset].iloc[-1])
                strike = st.number_input(f"قیمت اعمال پوت", key=f"{asset}_strike")
                premium = st.number_input(f"پریمیوم پوت", key=f"{asset}_premium")
                qty = st.number_input(f"تعداد دارایی", key=f"{asset}_qty", value=1)
                insurance_settings[asset] = {'price': price, 'strike': strike, 'premium': premium, 'qty': qty}

    # --------------------- محاسبه ریسک و بازده تعدیل‌شده ---------------------
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    adjusted_cov = cov_matrix.copy()
    for asset, setting in insurance_settings.items():
        insured_var = cov_matrix.loc[asset, asset]
        reduced_var = insured_var * 0.25  # کاهش ریسک به 25٪
        adjusted_cov.loc[asset, asset] = reduced_var

    # --------------------- پرتفو بهینه با ریسک تعدیل‌شده ---------------------
    st.subheader("🎯 پرتفو بهینه با ریسک تعدیل‌شده")
    target_risk = st.slider("هدف ریسک پرتفو (%)", 5, 50, 20) / 100
    num_assets = len(mean_returns)

    def portfolio_perf(weights, mean_returns, cov_matrix):
        returns = np.sum(mean_returns * weights)
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return returns, std

    def minimize_volatility(weights):
        return portfolio_perf(weights, mean_returns, adjusted_cov)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]

    result = minimize(minimize_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_weights = result.x

    risk, ret = portfolio_perf(opt_weights, mean_returns, adjusted_cov)
    st.write("ریسک پرتفو بهینه: {:.2f}٪".format(risk * 100))
    st.write("بازده مورد انتظار: {:.2f}٪".format(ret * 100))

    opt_df = pd.DataFrame({
        'دارایی': mean_returns.index,
        'وزن بهینه (%)': np.round(opt_weights * 100, 2)
    })
    st.dataframe(opt_df)

    # --------------------- نمودار Married Put ---------------------
    st.subheader("📊 نمودار سود و زیان استراتژی Married Put")
    for asset, setting in insurance_settings.items():
        current_price = setting['price']
        strike = setting['strike']
        premium = setting['premium']
        qty = setting['qty']

        prices = np.linspace(current_price * 0.7, current_price * 1.3, 200)
        payoff_stock = (prices - current_price) * qty
        payoff_put = np.where(prices < strike, (strike - prices), 0) * qty - premium * qty
        total_payoff = payoff_stock + payoff_put

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(prices, total_payoff, label="Married Put", color='blue')

        # ناحیه سود / زیان
        ax.fill_between(prices, total_payoff, where=(total_payoff >= 0), color='green', alpha=0.2)
        ax.fill_between(prices, total_payoff, where=(total_payoff < 0), color='red', alpha=0.2)

        # خط عمودی قیمت اعمال
        ax.axvline(x=strike, linestyle='--', color='black', label='Strike')

        # محور
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel("قیمت دارایی پایه")
        ax.set_ylabel("سود / زیان (P&L)")
        ax.set_title(f"📌 استراتژی Married Put برای دارایی: {asset}")
        ax.legend()
        st.pyplot(fig)
