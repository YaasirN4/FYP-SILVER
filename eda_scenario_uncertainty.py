import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from step7_scenario_system import interpret_scenario

print("Generating EDA Chart: Stochastic Scenario Forecast with Uncertainty Bands...")

# 1. Load Baseline Forecast
try:
    baseline = pd.read_csv("baseline_forecast.csv", index_col='Date', parse_dates=True)
    prices = baseline['Baseline_Forecast']
except FileNotFoundError:
    print("Error: baseline_forecast.csv not found. Please run step5_forecasting.py first.")
    exit(1)

# 2. Define a Test Scenario
user_scenario = "Global inflation is rising rapidly and geopolitical tensions are high."
print(f"Scenario: {user_scenario}")

# 3. Interpret Scenario
params = interpret_scenario(user_scenario)
trend_adj = params['trend_adj']
vol_adj = params['vol_adj']
explanation = params['explanation']

# 4. Monte Carlo Simulation (1000 paths for smooth bands)
np.random.seed(42)
days = len(prices)
num_paths = 1000
base_daily_vol = 0.015

# Calculate the mean trend line
trend_multipliers = np.linspace(1.0, 1.0 + trend_adj, days)
scenario_mean = prices * trend_multipliers

# Generate paths
paths = []
for _ in range(num_paths):
    shocks = np.random.normal(0, base_daily_vol * vol_adj, days)
    # Using cumulative returns for more realistic paths if needed, 
    # but for simplicity matching the app.py logic of daily shocks on mean
    paths.append(scenario_mean * (1 + shocks))

paths = np.array(paths)

# Calculate Percentiles for Uncertainty Bands
upper_bound = np.percentile(paths, 95, axis=0) # 95th percentile
lower_bound = np.percentile(paths, 5, axis=0)  # 5th percentile
median_path = np.percentile(paths, 50, axis=0) # Median path

# 5. Plotting
plt.figure(figsize=(14, 8), facecolor='#0d1117')
ax = plt.gca()
ax.set_facecolor('#0d1117')

# Plot Baseline
plt.plot(prices.index, prices, label='Baseline ARIMA Forecast', 
         color='#58a6ff', linestyle='--', linewidth=2, alpha=0.6)

# Plot Uncertainty Band
plt.fill_between(prices.index, lower_bound, upper_bound, 
                 color='#39d353', alpha=0.15, label='90% Confidence Interval (Uncertainty)')

# Plot Stochastic Paths (a few sample paths for texture)
for i in range(10):
    plt.plot(prices.index, paths[i], color='#39d353', alpha=0.05, linewidth=0.5)

# Plot Scenario Median/Actual Path
plt.plot(prices.index, median_path, label='Simulated Scenario Forecast', 
         color='#39d353', linewidth=3)

# Formatting
plt.title('Stochastic Simulation: Baseline vs. Scenario Adjusted Forecast\nScenario: ' + user_scenario[:50] + '...', 
          color='white', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', color='#8b949e', fontsize=12)
plt.ylabel('Silver Price (USD)', color='#8b949e', fontsize=12)
plt.xticks(color='#8b949e')
plt.yticks(color='#8b949e')
plt.grid(True, color='#21262d', linestyle='--', alpha=0.5)

# Legend
legend = plt.legend(facecolor='#161b22', edgecolor='#30363d', fontsize=11)
for text in legend.get_texts():
    text.set_color('white')

# Description Text
desc_text = f"Adjustment Factors:\nTrend: {trend_adj*100:+.1f}%\nVolatility: {vol_adj:.2f}x\n\nInterpretation:\n{explanation[:100]}..."
plt.text(0.02, 0.05, desc_text, transform=ax.transAxes, color='white', 
         fontsize=10, bbox=dict(facecolor='#161b22', alpha=0.8, edgecolor='#30363d'))

plt.tight_layout()
plt.savefig('silver_scenario_uncertainty.png', dpi=150, facecolor='#0d1117')
print("[DONE] Saved: silver_scenario_uncertainty.png")
plt.close()
