import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np

print("Generating EDA Chart: Moving Averages (50-day & 200-day)...")

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("cleaned_silver_prices.csv", index_col='Date', parse_dates=True)

# ── Calculate MAs ─────────────────────────────────────────────────────────────
df['MA_50']  = df['Close'].rolling(window=50).mean()
df['MA_200'] = df['Close'].rolling(window=200).mean()

# ── Detect Golden Cross / Death Cross ─────────────────────────────────────────
# Golden Cross: MA50 crosses above MA200 (bullish)
# Death Cross:  MA50 crosses below MA200 (bearish)
df['prev_MA50']  = df['MA_50'].shift(1)
df['prev_MA200'] = df['MA_200'].shift(1)
golden_crosses = df[
    (df['MA_50'] > df['MA_200']) & (df['prev_MA50'] <= df['prev_MA200'])
].index
death_crosses = df[
    (df['MA_50'] < df['MA_200']) & (df['prev_MA50'] >= df['prev_MA200'])
].index

# ── Figure Setup ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                          gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08},
                          facecolor='#0d1117')

ax1 = axes[0]   # Main price + MA chart
ax2 = axes[1]   # MA spread (momentum indicator)

for ax in [ax1, ax2]:
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#8b949e', labelsize=9)
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['right'].set_visible(False)

# ── Panel 1: Price + MAs ──────────────────────────────────────────────────────
ax1.fill_between(df.index, df['Close'], alpha=0.08, color='#58a6ff')
ax1.plot(df.index, df['Close'], color='#58a6ff', linewidth=1.0,
         label='Silver Close Price', alpha=0.85, zorder=2)
ax1.plot(df.index, df['MA_50'],  color='#ffa657', linewidth=1.8,
         label='50-Day MA  (Short-Term Momentum)', alpha=0.95, zorder=3)
ax1.plot(df.index, df['MA_200'], color='#ff7b72', linewidth=2.0,
         label='200-Day MA (Long-Term Trend)', linestyle='--', alpha=0.95, zorder=3)

# Shade bullish / bearish zones between the two MAs
valid = df[['MA_50', 'MA_200']].dropna()
ax1.fill_between(valid.index, valid['MA_50'], valid['MA_200'],
                 where=valid['MA_50'] >= valid['MA_200'],
                 alpha=0.12, color='#3fb950', label='Bullish Zone (MA50 > MA200)', zorder=1)
ax1.fill_between(valid.index, valid['MA_50'], valid['MA_200'],
                 where=valid['MA_50'] < valid['MA_200'],
                 alpha=0.12, color='#f85149', label='Bearish Zone (MA50 < MA200)', zorder=1)

# Mark Golden Crosses
for gc in golden_crosses:
    ax1.axvline(gc, color='#3fb950', linewidth=0.8, alpha=0.6, linestyle='--')
    ax1.scatter(gc, df.loc[gc, 'MA_50'], color='#3fb950', s=55, zorder=5,
                marker='^', edgecolors='#0d1117', linewidths=0.8)

# Mark Death Crosses
for dc in death_crosses:
    ax1.axvline(dc, color='#f85149', linewidth=0.8, alpha=0.6, linestyle='--')
    ax1.scatter(dc, df.loc[dc, 'MA_50'], color='#f85149', s=55, zorder=5,
                marker='v', edgecolors='#0d1117', linewidths=0.8)

# Annotate all-time high
ath_date = df['Close'].idxmax()
ath_val  = df['Close'].max()
ax1.annotate(f'  ATH: ${ath_val:.2f}\n  ({ath_date.strftime("%b %Y")})',
             xy=(ath_date, ath_val),
             xytext=(ath_date - pd.DateOffset(years=2), ath_val * 0.85),
             arrowprops=dict(arrowstyle='->', color='#e3b341', lw=1.4),
             color='#e3b341', fontsize=8.5, fontweight='bold')

ax1.set_ylabel('Price (USD / oz)', color='#8b949e', fontsize=10)
ax1.yaxis.set_label_position('left')
ax1.grid(True, color='#21262d', linewidth=0.6, linestyle='--', alpha=0.7)

# Title & subtitle
ax1.set_title('Silver Price — 50-Day & 200-Day Moving Averages\nMomentum & Long-Term Trend Analysis  |  2010 – 2025',
              color='#e6edf3', fontsize=13, fontweight='bold', pad=14, loc='left')

ax1.legend(loc='upper right', framealpha=0.15, facecolor='#161b22',
           edgecolor='#30363d', labelcolor='#e6edf3', fontsize=8.5)

# Cross legend annotation
gc_patch = mpatches.Patch(color='#3fb950', label='▲ Golden Cross (Bullish Signal)')
dc_patch = mpatches.Patch(color='#f85149', label='▼ Death Cross  (Bearish Signal)')
ax1.legend(handles=[
    plt.Line2D([0],[0], color='#58a6ff', lw=1.2, label='Close Price'),
    plt.Line2D([0],[0], color='#ffa657', lw=1.8, label='50-Day MA'),
    plt.Line2D([0],[0], color='#ff7b72', lw=2.0, linestyle='--', label='200-Day MA'),
    mpatches.Patch(color='#3fb950', alpha=0.5, label='Bullish Zone'),
    mpatches.Patch(color='#f85149', alpha=0.5, label='Bearish Zone'),
    plt.Line2D([0],[0], marker='^', color='w', markerfacecolor='#3fb950',
               markersize=8, label='Golden Cross (Bullish)', linestyle='None'),
    plt.Line2D([0],[0], marker='v', color='w', markerfacecolor='#f85149',
               markersize=8, label='Death Cross (Bearish)', linestyle='None'),
], loc='upper right', framealpha=0.25, facecolor='#161b22',
   edgecolor='#30363d', labelcolor='#e6edf3', fontsize=8)

# Hide x-axis on top panel
ax1.tick_params(labelbottom=False)

# ── Panel 2: MA Spread (Momentum Gap) ────────────────────────────────────────
spread = valid['MA_50'] - valid['MA_200']
ax2.fill_between(spread.index, spread, 0,
                 where=spread >= 0, alpha=0.45, color='#3fb950', label='Bullish')
ax2.fill_between(spread.index, spread, 0,
                 where=spread < 0,  alpha=0.45, color='#f85149', label='Bearish')
ax2.plot(spread.index, spread, color='#e6edf3', linewidth=0.7, alpha=0.5)
ax2.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')
ax2.set_ylabel('Spread\n(MA50 − MA200)', color='#8b949e', fontsize=8.5)
ax2.set_xlabel('Year', color='#8b949e', fontsize=10)
ax2.grid(True, color='#21262d', linewidth=0.5, linestyle='--', alpha=0.6)
ax2.legend(loc='upper right', framealpha=0.2, facecolor='#161b22',
           edgecolor='#30363d', labelcolor='#e6edf3', fontsize=8)

# Shared x-axis formatting
for ax in [ax1, ax2]:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center', color='#8b949e')

# ── Watermark / Footer ────────────────────────────────────────────────────────
fig.text(0.99, 0.01, 'FYP — Silver Price Forecasting Simulator',
         ha='right', va='bottom', fontsize=7.5, color='#484f58', style='italic')

plt.savefig('eda_moving_averages.png', dpi=180, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
print("[DONE] Saved: eda_moving_averages.png")
plt.show()
