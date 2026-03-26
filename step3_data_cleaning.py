import pandas as pd

# STEP 3: DATA CLEANING

print("Loading raw silver prices...")
# `index_col=0` tells pandas that the first column is the Date index
df = pd.read_csv("silver_prices.csv", header=[0, 1], index_col=0)

# 1. Convert Date index to datetime objects
df.index = pd.to_datetime(df.index)

# 2. Keep only the "Close" column
close_prices = df[[('Close', 'SI=F')]].copy()
close_prices.columns = ['Close']

# 3. Handle missing values
missing_before = close_prices['Close'].isna().sum()
print(f"Missing values found: {missing_before}")

close_prices = close_prices.dropna()

print("\n--- Cleaned Dataset Preview ---")
print(close_prices.head())

clean_csv_name = "cleaned_silver_prices.csv"
close_prices.to_csv(clean_csv_name)
print(f"\nSuccess! Cleaned data saved to '{clean_csv_name}'.")
