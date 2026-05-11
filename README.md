# 📈 Stock Predictor — ML Trading System

An end-to-end machine learning trading system built in Python that predicts daily stock price direction (UP/DOWN) for Indian stocks (NSE) using XGBoost.

## 🎯 Features
- Real-time data fetching via Yahoo Finance
- 9 technical indicators (RSI, Bollinger Bands, Moving Averages, etc.)
- XGBoost ML model with 54%+ accuracy
- Time-based train/test split (no future data leakage)
- Full backtesting engine with trailing stop loss
- Multi-stock screener tested on 5 NSE stocks
- Buy & Hold benchmark comparison
- Daily prediction report

## 📊 Results
Tested on 5 major NSE stocks (Reliance, TCS, Infosys, HDFC Bank, ICICI Bank):
- 4 out of 5 stocks beat buy & hold strategy
- Best performer: ICICI Bank with 36.4% return
- Average accuracy across stocks: ~50-55%

## 🛠️ Tech Stack
- Python 3.11
- pandas, numpy — data manipulation
- yfinance — stock data
- scikit-learn — ML utilities
- XGBoost — gradient boosting classifier
- matplotlib — visualization

## 🚀 How to Run
```bash
pip install -r requirements.txt
py test.py
py report.py
```

## ⚠️ Disclaimer
This is a learning project. Not financial advice. Do not trade real money based on this model.

## 👤 Author
**Rishit Vats** — Learning AI/ML through real projects