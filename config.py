from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("NO API KEY FOUND")
else:
    print("API KEY LOADED:", API_KEY[:12], "...")