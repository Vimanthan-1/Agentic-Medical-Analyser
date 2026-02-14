import google.generativeai as genai

# Setup
KEY = "AIzaSyDVJiMhsYRTUNgooGPEzEl2N26Vb3AMPvA" # Put your actual key inside these quotes
genai.configure(api_key=KEY)

# Select the fast, free model
model = genai.GenerativeModel('gemini-2.5-flash')

try:
    print("🤖 Contacting Google AI...")
    response = model.generate_content("Explain what Triage means in a hospital in one short sentence.")
    print("\n✅ SUCCESS! AI Responded:")
    print(response.text)
except Exception as e:
    print(f"\n❌ ERROR: {e}")