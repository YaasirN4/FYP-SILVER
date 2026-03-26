import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from textblob import TextBlob
import time

st.set_page_config(page_title="Silver Simulator AI", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

# 1. Custom CSS for a Premium Feel
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .stChatInputContainer {
        padding-bottom: 2rem;
    }
    h1 {
        font-family: 'Inter', sans-serif;
        background: -webkit-linear-gradient(#4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Silver Price Forecasting AI")
st.markdown("<p style='text-align: center; color: #888;'>Scenario-Based Simulator with NLP Sentiment Tracking & Stochastic ARIMA</p>", unsafe_allow_html=True)
st.markdown("---")

# 2. Advanced NLP Interpretation Engine
def advanced_interpret(scenario_text):
    text = scenario_text.lower()
    
    # Base params
    trend_adj = 0.0
    vol_adj = 1.0
    reasons = []
    
    # Sentiment Analysis (Fear/Positivity) using TextBlob
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity # -1.0 (negative) to 1.0 (positive)
    
    # Highly negative sentiment = Fear/Uncertainty -> Higher Volatility and slight price bump (Safe Haven)
    if sentiment < -0.1:
        vol_adj += abs(sentiment) * 2.0  # Scales up to +2.0x based on fear
        trend_adj += 0.02
        reasons.append(f"Negative sentiment detected ({sentiment:.2f}). Increasing volatility risk.")
    elif sentiment > 0.3:
        vol_adj *= 0.9 # Slight drop in volatility for positive outlook
        reasons.append(f"Positive sentiment detected ({sentiment:.2f}). Market stability limits volatility.")
        
    # Keyword Explicit Rules (Overrides or Adds)
    if "inflation" in text:
        trend_adj += 0.05
        vol_adj += 0.2
        reasons.append("Inflation keyword: Adding +5% trend. Silver acts as an inflation hedge.")
    if "usd" in text or "dollar" in text or "fed" in text:
        trend_adj -= 0.05
        reasons.append("USD keyword: Strong dollar natively pulls silver prices down (-5%).")
    if "demand" in text or "industry" in text:
        trend_adj += 0.10
        reasons.append("Industrial demand keyword: Adding +10% strong uptrend.")
    if "war" in text or "crisis" in text or "uncertainty" in text or "crash" in text:
        trend_adj += 0.08
        vol_adj += 1.0
        reasons.append("Crisis explicitly detected: Massive uncertainty triggers safe-haven buying (+8%).")
        
    if not reasons:
        reasons.append("No specific keywords. Relied purely on NLP Sentiment mathematical baseline.")
        
    # Cap parameters for realism
    trend_adj = max(min(trend_adj, 0.20), -0.20) # Max 20% swing
    vol_adj = max(min(vol_adj, 4.0), 0.5)        # 0.5x to 4.0x vol
    
    return trend_adj, vol_adj, reasons

# 3. Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! I am your AI Simulation Assistant. Type an economic scenario (e.g., *'The stock market just crashed and there is severe inflation.'*), and I will simulate how it impacts our Silver Price prediction!"}
    ]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "fig" in message:
            st.plotly_chart(message["fig"], use_container_width=True)

# 4. Load Data
@st.cache_data
def load_data():
    try:
        return pd.read_csv("baseline_forecast.csv", index_col='Date', parse_dates=True)
    except FileNotFoundError:
        return pd.DataFrame()

baseline_data = load_data()

# 5. Handle User Input
if prompt := st.chat_input("Type your economic event here (e.g., 'Inflation is out of control')"):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process AI Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🧠 *Analyzing natural language and calculating stochastic models...*")
        
        # Artificial delay to feel the AI "thinking"
        time.sleep(1.5)
        
        trend_adj, vol_adj, reasons = advanced_interpret(prompt)
        
        explanation = f"**Simulation Engine Results:**\n"
        explanation += "\n".join([f"- {r}" for r in reasons])
        explanation += f"\n\n**Calculated Adjustments:**\n- `Trend Trajectory`: {trend_adj*100:+.1f}%\n- `Volatility Factor`: {vol_adj:.2f}x"
        
        message_placeholder.markdown(explanation)
        
        if baseline_data.empty:
            st.error("Missing baseline_forecast.csv")
            st.stop()
            
        # 6. Generating Forecast with Confidence Intervals (Monte Carlo styling)
        prices = baseline_data['Baseline_Forecast']
        days = len(prices)
        
        # Base Trend Line
        trend_multipliers = np.linspace(1.0, 1.0 + trend_adj, days)
        scenario_mean = prices * trend_multipliers
        
        # Simulate 100 paths to build the shaded Confidence Interval
        np.random.seed(42)
        base_daily_vol = 0.015
        paths = []
        for _ in range(100):
            shocks = np.random.normal(0, base_daily_vol * vol_adj, days)
            paths.append(scenario_mean * (1 + shocks))
            
        paths = np.array(paths)
        upper_bound = np.percentile(paths, 95, axis=0) # 95th percentile bounds
        lower_bound = np.percentile(paths, 5, axis=0)  # 5th percentile bounds
        scenario_actual = paths[0] # The primary simulated line we show in solid
        
        # 7. Build Premium Interactive Plotly Chart
        fig = go.Figure()
        
        # Original Baseline
        fig.add_trace(go.Scatter(x=prices.index, y=prices,
                    mode='lines',
                    line=dict(color='rgba(100, 149, 237, 0.4)', width=2, dash='dash'),
                    name='ARIMA Baseline'))
                    
        # Confidence Interval Shading (Glassmorphism look)
        fig.add_trace(go.Scatter(x=prices.index.tolist() + prices.index[::-1].tolist(),
                    y=upper_bound.tolist() + lower_bound[::-1].tolist(),
                    fill='toself',
                    fillcolor='rgba(0, 242, 254, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='90% Scenario Confidence Interval',
                    hoverinfo='skip'))
                    
        # Primary Scenario Line
        fig.add_trace(go.Scatter(x=prices.index, y=scenario_actual,
                    mode='lines',
                    line=dict(color='#00f2fe', width=3),
                    name=f'Simulated Price (NLP Scenario)'))
                    
        fig.update_layout(
            title="Stochastic Scenario Forecasting",
            height=400,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(0,0,0,0.5)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Save to history so it doesn't disappear when they chat again
        st.session_state.messages.append({"role": "assistant", "content": explanation, "fig": fig})
