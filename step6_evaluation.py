import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

print("Generating Model Evaluation Chart: Actual vs. Predicted (60-Day Window)...")

# 1. Load data
df = pd.read_csv("cleaned_silver_prices.csv", index_col='Date', parse_dates=True)
prices = df['Close']

# 2. Split data into train/test (Last 60 days for testing)
test_size = 60
train = prices.iloc[:-test_size]
test = prices.iloc[-test_size:]

# 3. Train the model ONLY on the training data
print("Training ARIMA(5, 1, 0) on training set...")
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# 4. Generate forecast for the test period
print("Generating 60-day out-of-sample forecast...")
forecast = model_fit.forecast(steps=test_size)
forecast.index = test.index  # Align dates

# 5. Calculate Metrics
rmse = np.sqrt(np.mean((test - forecast)**2))
mae = np.mean(np.abs(test - forecast))
mape = np.mean(np.abs((test - forecast) / test)) * 100

# 6. Premium Visualization
plt.figure(figsize=(15, 8), facecolor='#0d1117')
ax = plt.gca()
ax.set_facecolor('#0d1117')

# Plot Training Context (Last 120 days of training)
plt.plot(train.index[-120:], train.iloc[-120:], color='#58a6ff', 
         label='Training Data (History)', alpha=0.5, linewidth=1.5)

# Plot Actual Test Data
plt.plot(test.index, test, color='#39d353', label='Actual Price (Ground Truth)', 
         linewidth=2.5, zorder=3)

# Plot ARIMA Forecast
plt.plot(forecast.index, forecast, color='#ff7b72', label='ARIMA Model Prediction', 
         linestyle='--', linewidth=2.5, zorder=4)

# Error Shading (Fill between actual and predicted)
plt.fill_between(test.index, test, forecast, color='#ff7b72', alpha=0.1, label='Prediction Error')

# Formatting
plt.title('ARIMA(5,1,0) Backtesting — Actual vs. Predicted\nTarget: 60-Day Testing Window', 
          color='#e6edf3', fontsize=16, fontweight='bold', pad=20, loc='left')

plt.xlabel('Date', color='#8b949e', fontsize=12)
plt.ylabel('Silver Price (USD / oz)', color='#8b949e', fontsize=12)
plt.xticks(color='#8b949e')
plt.yticks(color='#8b949e')

# Grid
plt.grid(True, color='#21262d', linestyle='--', alpha=0.6)

# Hide top/right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#30363d')
ax.spines['bottom'].set_color('#30363d')

# Metrics Box
metrics_text = (f"Model Metrics:\n"
                f"RMSE: ${rmse:.2f}\n"
                f"MAE:  ${mae:.2f}\n"
                f"MAPE: {mape:.1f}%")
plt.text(0.02, 0.95, metrics_text, transform=ax.transAxes, color='white', 
         fontsize=11, fontweight='bold', verticalalignment='top',
         bbox=dict(facecolor='#161b22', alpha=0.8, edgecolor='#30363d', boxstyle='round,pad=1'))

# Legend
plt.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', 
           fontsize=10, loc='lower right')

plt.tight_layout()
plt.savefig('silver_actual_vs_predicted.png', dpi=150, facecolor='#0d1117')
print(f"[DONE] Saved comparison chart to 'silver_actual_vs_predicted.png'")

# Save metrics to file
with open("evaluation_metrics.txt", "w") as f:
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"MAPE: {mape:.2f}%\n")

plt.close()
