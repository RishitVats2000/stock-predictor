import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import config

# Get data
df = yf.download("RELIANCE.NS", start=config.START_DATE, end=date.today(), progress=False)
df.columns = df.columns.get_level_values(0)

# ── Chart 1: Price + Moving Averages ──────────────
df['MA20'] = df['Close'].rolling(20).mean()
df['MA50'] = df['Close'].rolling(50).mean()

plt.figure(figsize=(14, 6))
plt.plot(df.index[-365:], df['Close'][-365:], label='Closing Price', alpha=0.8)
plt.plot(df.index[-365:], df['MA20'][-365:], label='20-Day MA', color='orange')
plt.plot(df.index[-365:], df['MA50'][-365:], label='50-Day MA', color='red')
plt.title("Reliance Industries — Price + Moving Averages (Last 1 Year)")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.tight_layout()
plt.savefig('chart_price_ma.png', dpi=100, bbox_inches='tight')
plt.close()
print("✅ Saved: chart_price_ma.png")

# ── Chart 2: Screener Results Bar Chart ──────────
df_results = pd.read_csv(config.SCREENER_FILE)

plt.figure(figsize=(14, 6))
colors = ['green' if x > 0 else 'red' for x in df_results['Model Return %']]
plt.bar(df_results['Stock'], df_results['Model Return %'], color=colors, alpha=0.7, label='Model')
plt.bar(df_results['Stock'], df_results['Buy & Hold %'], color='blue', alpha=0.3, label='Buy & Hold')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title("Stock Screener Results — Model vs Buy & Hold")
plt.ylabel("Return (%)")
plt.legend()
plt.tight_layout()
plt.savefig('chart_screener.png', dpi=100, bbox_inches='tight')
plt.close()
print("✅ Saved: chart_screener.png")

print("\nAll charts saved! Check the folder.")