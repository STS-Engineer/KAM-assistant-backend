import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
