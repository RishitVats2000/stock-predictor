# ── Trading Strategy Configuration ──────────────────

# Stocks to analyze
STOCKS = [
    ('RELIANCE.NS',  'Reliance'),
    ('TCS.NS',       'TCS'),
    ('INFY.NS',      'Infosys'),
    ('HDFCBANK.NS',  'HDFC Bank'),
    ('ICICIBANK.NS', 'ICICI Bank'),
]

# Time period
START_DATE = "2015-01-01"

# Model settings
TRAIN_TEST_SPLIT = 0.8        # 80% train, 20% test
N_ESTIMATORS     = 100        # number of trees

# Trading rules
STARTING_CAPITAL    = 100000  # ₹1,00,000
CONFIDENCE_THRESHOLD = 0.65   # min confidence to trade
STOP_LOSS_PERCENT    = 0.05   # 5% trailing stop loss

# Output files
SCREENER_FILE   = "stock_screener.csv"
PREDICTIONS_FILE = "predictions.csv"
REPORT_FILE     = "report.txt"