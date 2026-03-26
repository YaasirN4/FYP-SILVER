print("STEP 7: Scenario Interpretation System")
print("Loading scenario interpretation logic...\n")

def interpret_scenario(user_input):
    """
    Takes user text input, looks for keywords, and outputs
    how much the trend and volatility should change.
    """
    # Convert input to lowercase for easier matching
    text = user_input.lower()
    
    # Default baseline parameters
    # trend_adj is a percentage adjustment to our future baseline prices (e.g., 0.05 means +5%)
    # vol_adj increases the daily randomness of the price (e.g., 1.5 means 50% more volatile)
    params = {
        "trend_adj": 0.0, 
        "vol_adj": 1.0,
        "explanation": "No specific scenario detected. Using baseline model predictions."
    }
    
    if "inflation increases" in text or "high inflation" in text:
        # Silver is an inflation hedge; price goes up, slight volatility increase
        params["trend_adj"] = 0.05  
        params["vol_adj"] = 1.2     
        params["explanation"] = "Inflation detected. Silver is a known hedge against inflation. Adjusted baseline prices UP (+5%) and slightly increased volatility."
        
    elif "usd strengthens" in text or "strong dollar" in text:
        # Strong dollar makes silver more expensive for others, so price drops
        params["trend_adj"] = -0.05  
        params["vol_adj"] = 1.1      
        params["explanation"] = "Strong USD detected. Silver is priced in USD, so a strong dollar makes silver cheaper natively. Adjusted prices DOWN (-5%)."
        
    elif "demand rises" in text or "industry demand" in text:
        # Industrial demand pushes price up steadily
        params["trend_adj"] = 0.10  
        params["vol_adj"] = 1.0      
        params["explanation"] = "High demand detected. Strong industrial need will steadily drive prices UP (+10%)."
        
    elif "uncertainty increases" in text or "war" in text or "crisis" in text:
        # Silver is a safe haven in times of high crisis; wild price swings
        params["trend_adj"] = 0.08  
        params["vol_adj"] = 2.0      
        params["explanation"] = "Geopolitical uncertainty detected. Investors flock to precious metals. Adjusted prices UP (+8%) with HIGH volatility."
        
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
