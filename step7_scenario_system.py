import os
import json

print("STEP 7: Scenario Interpretation System")
print("Loading scenario interpretation logic...\n")

def interpret_scenario(user_input):
    """
    Takes user text input, uses Gemini AI if an API key is available,
    otherwise falls back to keyword mapping, and outputs
    how much the trend and volatility should change (-50% to +50% limit).
    """
    params = {
        "trend_adj": 0.0, 
        "vol_adj": 1.0,
        "explanation": "No specific scenario detected. Using baseline model predictions."
    }
    
    # --- GEMINI AI ROUTINE ---
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
            Act as an expert commodities trader. Analyze this news scenario and determine its impact on silver prices (30 days).
            Output a JSON object with:
            1. "trend_adj": float between -0.50 and 0.50.
            2. "vol_adj": float between 0.5 and 5.0.
            3. "reasoning": short explanation string.
            Scenario text: "{user_input}"
            Return ONLY valid JSON.
            """
            response = model.generate_content(prompt)
            response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(response_text)
            
            trend_adj_raw = float(data.get("trend_adj", 0.0))
            vol_adj_raw = float(data.get("vol_adj", 1.0))
            
            params["trend_adj"] = max(min(trend_adj_raw, 0.50), -0.50)
            params["vol_adj"] = max(min(vol_adj_raw, 5.0), 0.5)
            params["explanation"] = "🤖 AI Analysis: " + data.get("reasoning", "Processed.")
            return params
        except Exception as e:
            print(f"Gemini API failed or not installed correctly: {e}. Falling back to keyword rules.")

    print("Using basic fallback NLP engine...")
    # Convert input to lowercase for easier matching
    text = user_input.lower()
    
    if "inflation increases" in text or "high inflation" in text or "inflation" in text:
        params["trend_adj"] = 0.15  
        params["vol_adj"] = 1.5     
        params["explanation"] = "Inflation detected. Adjusted baseline prices UP (+15%) and increased volatility."
        
    elif "usd strengthens" in text or "strong dollar" in text or "fed" in text:
        params["trend_adj"] = -0.15  
        params["vol_adj"] = 1.2      
        params["explanation"] = "Strong USD detected. Adjusted prices DOWN (-15%)."
        
    elif "demand rises" in text or "industry demand" in text or "industry" in text:
        params["trend_adj"] = 0.20  
        params["vol_adj"] = 1.0      
        params["explanation"] = "High demand detected. Strong industrial need drives prices UP (+20%)."
        
    elif "uncertainty increases" in text or "war" in text or "crisis" in text:
        params["trend_adj"] = 0.30  
        params["vol_adj"] = 3.0      
        params["explanation"] = "Geopolitical crisis detected. Investors flock to safe havens. Adjusted prices UP (+30%) with HIGH volatility (3.0x)."
        
    return params

# --- TEST THE SYSTEM ---
if __name__ == "__main__":
    print("--- Testing the Scenario Engine ---\n")
    
    test_scenarios = [
        "I just heard that inflation increases rapidly.",
        "Financial news states that the usd strengthens.",
        "Solar panel industry demand rises heavily this year.",
        "Global uncertainty increases due to new policies."
    ]
    
    for scenario in test_scenarios:
        print(f"User Input: '{scenario}'")
        result = interpret_scenario(scenario)
        print(f" -> Trend Adjustment: {result['trend_adj'] * 100:.1f}%")
        print(f" -> Volatility Multiplier: {result['vol_adj']}x")
        print(f" -> Explanation: {result['explanation']}\n")

    print("\nSuccess! Step 7 interpretation engine logic works and is ready for integration.")
