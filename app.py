import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from textblob import TextBlob
import time
import json
import google.generativeai as genai

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

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ App Settings")
    st.markdown("Enter your free [Google Gemini API Key](https://aistudio.google.com/app/apikey) below to enable highly-advanced AI scenario parsing. Without it, the app will use basic keyword analysis.")
    api_key = st.text_input("Gemini API Key", type="password", help="Get a free key at aistudio.google.com")
    st.markdown("---")
    st.header("🏢 Business Procurement Options")
    st.markdown("Calculate real-world ROI based on the AI forecast.")
    req_ounces = st.number_input("Required Silver (Ounces)", min_value=0, max_value=1_000_000, value=50000, step=1000)
    current_spot_price = st.number_input("Proxy Spot Price ($/oz)", min_value=10.0, max_value=150.0, value=28.5, step=0.1)

    st.markdown("---")
    st.caption("**Market Sensitivity Config:** High")
    
    if api_key:
        genai.configure(api_key=api_key)
        st.success("Gemini API Enabled!")
    else:
        st.warning("Using Fallback NLP (Keyword-based)")

# 2. Advanced NLP Interpretation Engine
def advanced_interpret(scenario_text, use_gemini=False):
    # Base fallback params
    trend_adj = 0.0
    vol_adj = 1.0
    reasons = []
    
    # --- GEMINI AI ROUTINE ---
    if use_gemini:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
            Act as an expert commodities trader and quantitative analyst. Analyze the following news scenario and determine its theoretical impact on silver prices over the next 30 days.
            Silver is a safe-haven asset, an inflation hedge, and heavily used in industry (solar panels, electronics). It trades inversely to the USD.
            
            We are operating a HIGH SENSITIVITY model. Major global crises or massive industrial booms should cause extreme swings.
            Output your findings as a strict JSON object with exactly three keys:
            1. "trend_adj": A float representing the % price change trajectory. e.g., 0.15 (+15%), -0.25 (-25%), 0.50 (+50%). Min is -0.50, Max is 0.50. Be aggressive if the news warrants it.
            2. "vol_adj": A float representing the volatility multiplier. 1.0 is normal. 2.0 is double volatility. 4.0 is extreme chaos/panic. Min 0.5, Max 5.0.
            3. "reasoning": A single short string explaining your quantitative choices in aggressive trading terms.
            
            Scenario text: "{scenario_text}"
            
            Return ONLY the valid JSON object. No markdown, no "```json", just the raw JSON.
            """
            
            response = model.generate_content(prompt)
            # Clean up potential markdown formatting from the response
            response_text = response.text.strip()
            # Robust JSON extraction to prevent markdown crashes
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                response_text = response_text[start_idx:end_idx+1]

            
            data = json.loads(response_text)
            
            trend_adj = float(data.get("trend_adj", 0.0))
            vol_adj = float(data.get("vol_adj", 1.0))
            reason_str = list((data.get("reasoning", "Gemini processed the text but provided no reason.")).split('\n'))
            
            # Ensure bounds are respected even if Gemini goes rogue
            trend_adj = max(min(trend_adj, 0.50), -0.50) # Extremely High Sensitivity allowed
            vol_adj = max(min(vol_adj, 5.0), 0.5)
            
            reason_str[0] = "🤖 **Gemini AI Analysis:** " + reason_str[0]
            return trend_adj, vol_adj, reason_str
            
        except Exception as e:
            reasons.append(f"⚠️ Gemini API failed: {e}. Falling back to basic NLP.")
            # Fall through to TextBlob logic...

    # --- FALLBACK TEXTBLOB/KEYWORD ROUTINE ---
    text = scenario_text.lower()
    
    # Sentiment Analysis (Fear/Positivity) using TextBlob
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity # -1.0 (negative) to 1.0 (positive)
    
    # Highly negative sentiment = Fear/Uncertainty -> Higher Volatility and slight price bump (Safe Haven)
    if sentiment < -0.1:
        vol_adj += abs(sentiment) * 3.0  # Scales up aggressively
        trend_adj += 0.05
        reasons.append(f"Negative sentiment detected ({sentiment:.2f}). Escalating volatility risk heavily.")
    elif sentiment > 0.3:
        vol_adj *= 0.8
        reasons.append(f"Positive sentiment detected ({sentiment:.2f}). Reducing volatility factor.")
        
    # Keyword Explicit Rules (High Sensitivity overrides)
    if "inflation" in text:
        trend_adj += 0.15
        vol_adj += 0.5
        reasons.append("Inflation flag: Pricing in strong +15% aggressive upside scenario.")
    if "usd" in text or "dollar" in text or "fed" in text:
        trend_adj -= 0.15
        reasons.append("USD strength flag: Pricing in steep -15% commodity selloff.")
    if "demand" in text or "industry" in text:
        trend_adj += 0.20
        reasons.append("Industrial demand flag: Supercycle priced in, appending +20% run-up.")
    if "war" in text or "crisis" in text or "crash" in text:
        trend_adj += 0.30
        vol_adj += 3.0
        reasons.append("Crisis explicitly triggered: Max fear cycle initiated, massive +30% safe-haven premium.")
        
    if not reasons:
        reasons.append("No specific keywords. Baseline sentiment adjustments applied.")
        
    # Cap parameters for high sensitivity realism
    trend_adj = max(min(trend_adj, 0.50), -0.50) # Expanded from 20% to 50%
    vol_adj = max(min(vol_adj, 5.0), 0.5)        
    
    return trend_adj, vol_adj, reasons

# 3. Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! I am your AI Simulation Assistant. Type an economic scenario (e.g., *'The stock market just crashed and there is severe inflation.'*), and I will simulate how it impacts our Silver Price prediction!"}
    ]

# Display Chat History
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "fig" in message:
            st.plotly_chart(message["fig"], use_container_width=True, key=f"chart_{i}")
        if "proc_txt" in message:
            st.markdown(message["proc_txt"])

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
        
        has_key = True if api_key else False
        trend_adj, vol_adj, reasons = advanced_interpret(prompt, use_gemini=has_key)
        
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
        
        st.plotly_chart(fig, use_container_width=True, key=f"chart_active_{len(st.session_state.messages)}")
        
        # 8. Procurement ROI Engine
        current_cost = req_ounces * current_spot_price
        # Predict end-of-month price
        future_predicted_price = current_spot_price * (1 + trend_adj)
        future_cost = req_ounces * future_predicted_price
        savings = current_cost - future_cost 
        
        procurement_summary = f"### 🏢 Procurement Impact Engine\n"
        procurement_summary += f"**Scenario Impact on {req_ounces:,} oz Purchase Requirement:**\n"
        if trend_adj > 0:
            procurement_summary += f"\n> ⚠️ **BUY NOW RECOMMENDATION:** The model forecasts an aggressive price surge. Purchasing today avoids an estimated **${abs(savings):,.2f}** in extra future costs."
        elif trend_adj < 0:
            procurement_summary += f"\n> ✅ **WAIT RECOMMENDATION:** The model forecasts a price drop. Delaying your purchase by 30 days will save your business an estimated **${abs(savings):,.2f}**."
        else:
            procurement_summary += f"\n> ⚖️ **NEUTRAL RECOMMENDATION:** Prices are stable. Buy according to your normal schedule."
            
        st.info(procurement_summary)
        
        # Save to history so it doesn't disappear when they chat again
        st.session_state.messages.append({"role": "assistant", "content": explanation, "fig": fig, "proc_txt": procurement_summary})
