import os
import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyC1Zu2DMYQiAVxB0TGzaen3ISw7TMSKW0o")
genai.configure(api_key=GEMINI_API_KEY)

print("Available models:")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
