import os
import requests
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

response = requests.get(
    'https://api.groq.com/openai/v1/models',
    headers={'Authorization': f'Bearer {GROQ_API_KEY}'}
)
models = response.json().get('data', [])
for m in models:
    print(m['id'])
