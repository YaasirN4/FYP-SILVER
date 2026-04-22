import yfinance as yf
import pandas as pd

print("Downloading Silver Price Data (Ticker: SI=F)...")
# Download data
silver_data = yf.download("SI=F", start="2010-01-01")

print("\n--- First 5 rows of the dataset ---")
print(silver_data.head())

# Save dataset to CSV
csv_filename = "silver_prices.csv"
silver_data.to_csv(csv_filename)
print(f"\nSuccess! Data saved to {csv_filename}")
