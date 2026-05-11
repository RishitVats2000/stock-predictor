import yfinance as yf
import pandas as pd
from datetime import date
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import config   # ← import our settings!

# ── Helper: Test ONE stock ───────────────────────────
def test_stock(ticker, name):
    print(f"Testing {name} ({ticker})...")
    
    df = yf.download(ticker, start=config.START_DATE, end=date.today(), progress=False)
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
    split = int(len(X) * config.TRAIN_TEST_SPLIT)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train
    model = XGBClassifier(n_estimators=config.N_ESTIMATORS, 
                          random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Backtest
    capital = config.STARTING_CAPITAL
    cash, shares, trades = capital, 0, 0
    first_price = df['Close'].iloc[split]
    
    for i in range(len(X_test)):
        price = df['Close'].iloc[split + i]
        pred  = predictions[i]
        conf  = max(probs[i])
        
        if conf > config.CONFIDENCE_THRESHOLD and pred == 1 and cash >= price:
            shares_bought = int(cash / price)
            cash -= shares_bought * price
            shares += shares_bought
            trades += 1
        elif conf > config.CONFIDENCE_THRESHOLD and pred == 0 and shares > 0:
            cash += shares * price
            shares = 0
            trades += 1
    
    final_value     = cash + shares * df['Close'].iloc[-1]
    model_return    = round((final_value - capital) / capital * 100, 2)
    buyhold_return  = round((df['Close'].iloc[-1] / first_price - 1) * 100, 2)
    
    # Tomorrow's prediction
    latest = X.iloc[-1].values.reshape(1, -1)
    tomorrow_pred = model.predict(latest)[0]
    tomorrow_conf = max(model.predict_proba(latest)[0])
    
    return {
        'Stock': name,
        'Ticker': ticker,
        'Accuracy %': round(accuracy * 100, 2),
        'Model Return %': model_return,
        'Buy & Hold %': buyhold_return,
        'Beats Market': 'YES' if model_return > buyhold_return else 'NO',
        'Trades': trades,
        'Tomorrow': 'UP' if tomorrow_pred == 1 else 'DOWN',
        'Confidence %': round(tomorrow_conf * 100, 1),
        'Current Price': round(df['Close'].iloc[-1], 2),
        'Date': date.today()
    }

# ── Run Screener ────────────────────────────────────
print(f"\n{'='*60}")
print(f"  STOCK SCREENER — {date.today()}")
print(f"{'='*60}\n")

results = []
for ticker, name in config.STOCKS:
    results.append(test_stock(ticker, name))

# ── Save & Display ──────────────────────────────────
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('Model Return %', ascending=False)
df_results.to_csv(config.SCREENER_FILE, index=False)

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(df_results.to_string(index=False))
print(f"{'='*60}")
print(f"\n✅ Saved to {config.SCREENER_FILE}")