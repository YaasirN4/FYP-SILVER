import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("STEP 4: Exploratory Data Analysis")

# 1. Load cleaned data
print("Loading cleaned data...")
df = pd.read_csv("cleaned_silver_prices.csv", index_col='Date', parse_dates=True)

# 2. Add moving averages
print("Calculating Moving Averages (50-day and 200-day)...")
df['MA_50'] = df['Close'].rolling(window=50).mean()
df['MA_200'] = df['Close'].rolling(window=200).mean()

# Plot Price Trend and Moving Averages
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Silver Price (Close)', alpha=0.5)
plt.plot(df.index, df['MA_50'], label='50-Day Moving Average', color='orange')
plt.plot(df.index, df['MA_200'], label='200-Day Moving Average', color='red')
plt.title('Silver Price Trend with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('silver_price_trend.png')
print("Saved 'silver_price_trend.png'")
plt.close()

# 3. Show volatility (Rolling Standard Deviation)
print("Calculating 30-day Rolling Volatility...")
# Using 30-day rolling standard deviation of daily returns
df['Daily_Return'] = df['Close'].pct_change()
df['Volatility_30'] = df['Daily_Return'].rolling(window=30).std() * np.sqrt(252) # Annualized

# Plot Volatility
plt.figure(figsize=(12, 4))
plt.plot(df.index, df['Volatility_30'], label='30-Day Annualized Volatility', color='purple')
plt.title('Silver Price Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('silver_volatility.png')
print("Saved 'silver_volatility.png'")
plt.close()

# Save new features
df.to_csv("silver_prices_with_features.csv")
print("Saved dataset with calculated features to 'silver_prices_with_features.csv'")
