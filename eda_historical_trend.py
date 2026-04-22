import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print("Generating Historical Daily Silver Price Trend Chart (2010 - 2025)...")

# 1. Load Data
df = pd.read_csv("cleaned_silver_prices.csv", index_col='Date', parse_dates=True)
prices = df['Close']

# 2. Premium Visualization Setup
plt.figure(figsize=(16, 8), facecolor='#0d1117')
ax = plt.gca()
ax.set_facecolor('#0d1117')

# Plot the main trend
plt.plot(df.index, df['Close'], color='#58a6ff', linewidth=1.5, alpha=0.9, label='Daily Close Price')

# Add a smooth trend line (e.g., 200-day EMA for visual clarity)
ema_200 = df['Close'].ewm(span=200, adjust=False).mean()
plt.plot(df.index, ema_200, color='#ff7b72', linewidth=1.2, alpha=0.6, label='200-Day EMA Trend')

# Highlight All-Time High
ath_date = df['Close'].idxmax()
ath_val = df['Close'].max()
plt.scatter(ath_date, ath_val, color='#e3b341', s=100, zorder=5, edgecolors='white', label=f'All-Time High (${ath_val:.2f})')
plt.annotate(f'ATH: ${ath_val:.2f}\n({ath_date.strftime("%b %Y")})', 
             xy=(ath_date, ath_val), xytext=(ath_date + pd.DateOffset(months=6), ath_val + 5),
             arrowprops=dict(arrowstyle='->', color='#e3b341', lw=1.5),
             color='#e3b341', fontsize=10, fontweight='bold')

# Formatting
plt.title('Historical Daily Silver Price Trend\nDataset Duration: Jan 2010 – Apr 2025', 
          color='#e6edf3', fontsize=18, fontweight='bold', pad=25, loc='left')

plt.xlabel('Year', color='#8b949e', fontsize=12)
plt.ylabel('Price (USD / oz)', color='#8b949e', fontsize=12)
plt.xticks(color='#8b949e')
plt.yticks(color='#8b949e')

# Grid and Spines
plt.grid(True, color='#21262d', linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#30363d')
ax.spines['bottom'].set_color('#30363d')

# Legend
plt.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=11, loc='upper left')

# Date formatting
ax.xaxis.set_major_locator(mdates.YearLocator(2)) # Show every 2 years
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('silver_historical_trend_full.png', dpi=150, facecolor='#0d1117')
print("[DONE] Saved: silver_historical_trend_full.png")
plt.close()
