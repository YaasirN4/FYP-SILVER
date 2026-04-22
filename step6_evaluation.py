import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

print("STEP 6: Model Evaluation")

# 1. Load data
df = pd.read_csv("cleaned_silver_prices.csv", index_col='Date', parse_dates=True)
prices = df['Close']

# 2. Split data into train/test
# We use the last 60 days as the "test" set to evaluate model performance
test_size = 60
train = prices.iloc[:-test_size]
test = prices.iloc[-test_size:]

print(f"Total data points: {len(prices)}")
print(f"Training data points: {len(train)}")
print(f"Testing data points (The 'Future' we hide from the model): {len(test)}")

# 3. Train the model ONLY on the training data
print("\nTraining ARIMA model on the training set...")
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# 4. Generate forecast for the test period
print("Forecasting the test period...")
forecast = model_fit.forecast(steps=test_size)
forecast.index = test.index  # Align dates

# 5. Calculate RMSE, MAE, and MAPE
# RMSE (Root Mean Squared Error): Typical error in price dollars (punishes large errors)
rmse = np.sqrt(np.mean((test - forecast)**2))

# MAE (Mean Absolute Error): Average absolute difference in dollars
mae = np.mean(np.abs(test - forecast))

# MAPE (Mean Absolute Percentage Error): Error as a percentage
mape = np.mean(np.abs((test - forecast) / test)) * 100

print("\n--- Evaluation Results ---")
print(f"RMSE (Root Mean Sq Error): ${rmse:.2f}")
print(f"MAE  (Mean Absolute Error): ${mae:.2f}")
print(f"MAPE (Mean Abs % Error):   {mape:.2f}%")

print("\n--- What does this mean? ---")
print(f"Your model's predictions are, on average, off by ${rmse:.2f} compared to the actual price.")
print(f"In percentage terms, the model's predictions differ from actual prices by about {mape:.2f}%.")
# A MAPE under 10% is usually considered excellent forecasting.

with open("evaluation_metrics.txt", "w") as f:
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"MAPE: {mape:.2f}\n")

# 6. Plot the results to visually see Train vs Test vs Forecast
plt.figure(figsize=(12, 6))
# Plot last 200 days of train for context
plt.plot(train.index[-200:], train.iloc[-200:], label='Training Data')
plt.plot(test.index, test, label='Actual Test Data (Hidden from model)', color='green')
plt.plot(forecast.index, forecast, label='Model Forecast', color='red', linestyle='dashed', linewidth=2)

plt.title(f'ARIMA Performance on Test Data (MAPE: {mape:.2f}%)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('silver_evaluation.png')
print("\nSuccess! Saved evaluation graph to 'silver_evaluation.png'")
