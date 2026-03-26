import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Ignore statistical warnings for a cleaner output
warnings.filterwarnings("ignore")

print("STEP 5: Forecasting Model (ARIMA)")

# 1. Load data
print("Loading cleaned dataset...")
df = pd.read_csv("cleaned_silver_prices.csv", index_col='Date', parse_dates=True)
prices = df['Close']

# 2. Build and Train the Model
print("Training ARIMA(5, 1, 0) Model...")
# We use a standard baseline configuration for time-series
model = ARIMA(prices, order=(5, 1, 0))
model_fit = model.fit()

print("\n--- Model Training Summary ---")
# Print a simple part of the summary so it's not overwhelming
print(model_fit.summary().tables[1])

# 3. Generate future predictions (next 30 days)
print("\nGenerating 30-day forecast...")
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)

# Create future dates for the forecast (assuming trading/business days)
last_date = prices.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='B')
forecast.index = future_dates

print("\n--- Next 5 days forecast ---")
print(forecast.head())

# Save forecast to CSV
forecast.name = "Baseline_Forecast"
forecast.to_csv("baseline_forecast.csv", index_label='Date')
print("\nSuccess! Saved baseline forecast to 'baseline_forecast.csv'")

# 4. Plot historical data + forecast
plt.figure(figsize=(12, 6))
# Plot only the last 500 days of data so the forecast line is more visible
plt.plot(prices.index[-500:], prices.iloc[-500:], label='Historical Price (Last 500 Days)')
plt.plot(forecast.index, forecast, label='30-Day Forecast', color='red', linestyle='dashed', linewidth=2)
plt.title('Silver Price - Baseline ARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('silver_baseline_forecast.png')
print("Success! Saved forecast graph to 'silver_baseline_forecast.png'")
plt.close()
