import pandas as pd
import config
from datetime import date

# Load the screener results
df = pd.read_csv(config.SCREENER_FILE)

# Build the report
lines = []
lines.append("="*60)
lines.append(f"  STOCK ANALYSIS REPORT")
lines.append(f"  Generated: {date.today()}")
lines.append("="*60)
lines.append("")

# Section 1: Best Performer
best = df.iloc[0]
lines.append("🏆 BEST PERFORMER")
lines.append(f"   {best['Stock']} delivered {best['Model Return %']}% return")
lines.append(f"   Beat buy & hold by {round(best['Model Return %'] - best['Buy & Hold %'], 2)}%")
lines.append("")

# Section 2: All Stocks Summary
lines.append("📊 ALL STOCKS")
lines.append("-"*60)
for _, row in df.iterrows():
    status = "✅" if row['Beats Market'] == 'YES' else "❌"
    lines.append(f"{status} {row['Stock']:<12} | Return: {row['Model Return %']:>7}% | Tomorrow: {row['Tomorrow']} ({row['Confidence %']}%)")
lines.append("")

# Section 3: Trading Signals
lines.append("🎯 TOMORROW'S HIGH-CONFIDENCE SIGNALS")
lines.append("-"*60)
high_conf = df[df['Confidence %'] >= 60].sort_values('Confidence %', ascending=False)
if len(high_conf) > 0:
    for _, row in high_conf.iterrows():
        lines.append(f"   {row['Tomorrow']} signal on {row['Stock']} @ ₹{row['Current Price']} | Confidence: {row['Confidence %']}%")
else:
    lines.append("   No high-confidence signals today.")
lines.append("")

# Section 4: Summary Stats
lines.append("📈 STATISTICS")
lines.append("-"*60)
lines.append(f"   Average Accuracy:    {round(df['Accuracy %'].mean(), 2)}%")
lines.append(f"   Average Return:      {round(df['Model Return %'].mean(), 2)}%")
lines.append(f"   Stocks Beat Market:  {(df['Beats Market'] == 'YES').sum()} of {len(df)}")
lines.append(f"   Total Trades Made:   {df['Trades'].sum()}")
lines.append("")
lines.append("="*60)
lines.append("⚠️  This is a learning project. NOT financial advice.")
lines.append("="*60)

# Save report
report_text = "\n".join(lines)
with open(config.REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write(report_text)

# Print to terminal too
print(report_text)
print(f"\n✅ Report saved to {config.REPORT_FILE}")