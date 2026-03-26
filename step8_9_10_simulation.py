import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from step7_scenario_system import interpret_scenario

print("STEPS 8, 9, 10: Simulation Engine, Visualization, and Explanation")

# 1. Load Baseline Forecast
baseline = pd.read_csv("baseline_forecast.csv", index_col='Date', parse_dates=True)
prices = baseline['Baseline_Forecast']

# 2. STEP 8: Simulation Engine
# For this example, let's pretend the user submitted this scenario:
user_scenario = "Global uncertainty increases rapidly."
print(f"\nUser Scenario: '{user_scenario}'")

# Get parameters from Step 7
params = interpret_scenario(user_scenario)
trend_adj = params['trend_adj']
vol_adj = params['vol_adj']
explanation = params['explanation']

# Create Scenario Forecast
scenario_forecast = prices.copy()

# A. Apply Trend: Gradually apply the trend over the 30 days. 
# If trend is +5%, it reaches +5% on the final day.
days = len(prices)
trend_multipliers = np.linspace(1.0, 1.0 + trend_adj, days)
scenario_forecast = scenario_forecast * trend_multipliers

# B. Apply Volatility (Randomness)
# Historical daily volatility of silver is around 1.5% (0.015)
# We will generate daily random price shocks, magnified by vol_adj
np.random.seed(42) # For reproducible results
base_daily_volatility = 0.015  
daily_shocks = np.random.normal(0, base_daily_volatility * vol_adj, days)

# Apply shocks (1 + shock)
scenario_forecast = scenario_forecast * (1 + daily_shocks)

# 3. STEP 9: Visualization
plt.figure(figsize=(10, 5))
plt.plot(prices.index, prices, label='Baseline ARIMA Forecast (No Scenario)', color='blue', linestyle='dashed')
plt.plot(scenario_forecast.index, scenario_forecast, label=f"Scenario ({trend_adj*100}% trend, {vol_adj}x vol)", color='orange', linewidth=2)

plt.title('Baseline vs. Scenario Forecast')
plt.xlabel('Date')
plt.ylabel('Silver Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('silver_simulation_comparison.png')
print("\nSaved comparison graph to 'silver_simulation_comparison.png' (STEP 9)")

# 4. STEP 10: Explanation Generator
print("\n--- STEP 10: Explanation Generator ---")
print("Here is the explanation for the user:")
print(f"[{explanation}]")

df_results = pd.DataFrame({
    'Baseline': prices,
    'Scenario': scenario_forecast
})
df_results.to_csv("scenario_results.csv")
print("\nSuccess! Steps 8, 9, 10 are complete.")
