import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

print("Generating Premium Baseline Forecast Chart...")

# 1. Load Data
df = pd.read_csv("cleaned_silver_prices.csv", index_col='Date', parse_dates=True)
prices = df['Close']

# 2. Re-train the full model for the best forecast
model = ARIMA(prices, order=(5, 1, 0))
model_fit = model.fit()

# 3. Forecast 30 days
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)
future_dates = pd.date_range(start=prices.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')
forecast.index = future_dates

# 4. Premium Visualization
plt.figure(figsize=(15, 8), facecolor='#0d1117')
ax = plt.gca()
ax.set_facecolor('#0d1117')

# Plot History (Last 250 days for focus)
plt.plot(prices.index[-250:], prices.iloc[-250:], color='#58a6ff', 
         label='Historical Silver Price', linewidth=2, alpha=0.8)

# Plot Forecast
plt.plot(forecast.index, forecast, color='#ff7b72', label='ARIMA(5,1,0) Baseline Forecast', 
         linestyle='--', linewidth=3, marker='o', markersize=4)

# Formatting
plt.title('Baseline 30-Day Silver Price Forecast\nStatistical Projection (No External Scenarios Applied)', 
          color='#e6edf3', fontsize=16, fontweight='bold', pad=20, loc='left')

plt.xlabel('Date', color='#8b949e', fontsize=12)
plt.ylabel('Price (USD / oz)', color='#8b949e', fontsize=12)
plt.xticks(color='#8b949e')
plt.yticks(color='#8b949e')

# Grid
plt.grid(True, color='#21262d', linestyle='--', alpha=0.5)

# Legend
plt.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=11, loc='upper left')

# Spines
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Date formatting
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.tight_layout()
plt.savefig('silver_baseline_premium.png', dpi=150, facecolor='#0d1117')
print("[DONE] Saved: silver_baseline_premium.png")
plt.close()
