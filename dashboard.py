import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import config

# ── Page setup ──────────────────────────────────────
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

st.title("📈 Stock Predictor — ML Trading Dashboard")
st.markdown("*Powered by XGBoost on real NSE data*")
st.divider()

# ── Sidebar ─────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

stock_options = {name: ticker for ticker, name in config.STOCKS}
selected_name = st.sidebar.selectbox("Select a stock:", list(stock_options.keys()))
selected_ticker = stock_options[selected_name]

confidence_threshold = st.sidebar.slider(
    "Confidence threshold (%)", 50, 90, 65) / 100

starting_capital = st.sidebar.number_input(
    "Starting Capital (₹)", min_value=10000, value=100000, step=10000)

run_button = st.sidebar.button("🚀 Run Analysis", type="primary")

# ── Main Analysis ───────────────────────────────────
if run_button:
    with st.spinner(f"Analyzing {selected_name}..."):
        # Get data
        df = yf.download(selected_ticker, start=config.START_DATE, 
                         end=date.today(), progress=False)
        df.columns = df.columns.get_level_values(0)
        
        # Features
        df['MA5_ratio']    = df['Close'] / df['Close'].rolling(5).mean()
        df['MA20_ratio']   = df['Close'] / df['Close'].rolling(20).mean()
        df['MA50_ratio']   = df['Close'] / df['Close'].rolling(50).mean()
        df['Daily Return'] = df['Close'].pct_change()
        df['Volume Change']= df['Volume'].pct_change()
        df['Price Range']  = (df['High'] - df['Low']) / df['Close']
        df['Return_5d']    = df['Close'].pct_change(5)
        
        delta = df['Close'].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = -delta.where(delta < 0, 0).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / loss))
        df['RSI'] = df['RSI'].replace([float('inf'), float('-inf')], 50)
        
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.replace([float('inf'), float('-inf')], float('nan'))
        df.dropna(inplace=True)
        
        features = ['MA5_ratio', 'MA20_ratio', 'MA50_ratio',
                    'Daily Return', 'Volume Change',
                    'Price Range', 'Return_5d', 'RSI']
        
        X = df[features]
        y = df['Target']
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train
        model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probs = model.predict_proba(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Backtest
        cash, shares, trades = starting_capital, 0, 0
        first_price = df['Close'].iloc[split]
        port_values = []
        
        for i in range(len(X_test)):
            price = df['Close'].iloc[split + i]
            pred  = predictions[i]
            conf  = max(probs[i])
            
            if conf > confidence_threshold and pred == 1 and cash >= price:
                shares_bought = int(cash / price)
                cash -= shares_bought * price
                shares += shares_bought
                trades += 1
            elif conf > confidence_threshold and pred == 0 and shares > 0:
                cash += shares * price
                shares = 0
                trades += 1
            port_values.append(cash + shares * price)
        
        final_value = cash + shares * df['Close'].iloc[-1]
        model_return = (final_value - starting_capital) / starting_capital * 100
        buyhold_return = (df['Close'].iloc[-1] / first_price - 1) * 100
        
        # Tomorrow's prediction
        latest = X.iloc[-1].values.reshape(1, -1)
        tomorrow_pred = model.predict(latest)[0]
        tomorrow_conf = max(model.predict_proba(latest)[0])
        
    # ── Display Results ───────────────────────────
    st.success("Analysis complete! ✅")
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
    col2.metric("Model Accuracy", f"{accuracy*100:.1f}%")
    col3.metric("Tomorrow Says", "📈 UP" if tomorrow_pred == 1 else "📉 DOWN")
    col4.metric("Confidence", f"{tomorrow_conf*100:.1f}%")
    
    st.divider()
    
    # Backtest results
    st.subheader("📊 Backtest Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Your Strategy", f"{model_return:.2f}%", 
                delta=f"{model_return - buyhold_return:.2f}% vs market")
    col2.metric("Buy & Hold", f"{buyhold_return:.2f}%")
    col3.metric("Total Trades", trades)
    
    # Portfolio chart
    st.subheader("💼 Portfolio Growth")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(port_values, color='green', label='Strategy')
    ax.axhline(y=starting_capital, color='red', linestyle='--', label='Starting Capital')
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Value (₹)")
    ax.legend()
    st.pyplot(fig)
    
    # Price chart
    st.subheader(f"📈 {selected_name} — Last 1 Year")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    last_year = df.tail(252)
    ax2.plot(last_year.index, last_year['Close'], label='Close Price', color='blue')
    ax2.plot(last_year.index, last_year['Close'].rolling(20).mean(), 
             label='20-day MA', color='orange')
    ax2.plot(last_year.index, last_year['Close'].rolling(50).mean(), 
             label='50-day MA', color='red')
    ax2.legend()
    ax2.set_ylabel("Price (INR)")
    st.pyplot(fig2)
    
    # Disclaimer
    st.warning("⚠️ This is a learning project. Not financial advice.")
else:
    st.info("👈 Configure settings in the sidebar and click **Run Analysis** to begin!")
    st.markdown("""
    ### 🎯 What This Dashboard Does:
    - Trains an XGBoost ML model on 10 years of NSE stock data
    - Backtests a trading strategy with your chosen confidence threshold
    - Predicts tomorrow's price direction
    - Compares your strategy vs simple buy-and-hold
    
    Select a stock and click **Run Analysis** to see the magic! ✨
    """)