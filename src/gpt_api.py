import openai
import httpx
import time
import os

# Load your API key
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
if openai.api_key is None:
    raise ValueError("API key not found. Set the OPENAI_API_KEY environment variable.")

client = openai.OpenAI(
    api_key=openai.api_key,
    base_url=openai.base_url,
    http_client = httpx.Client(
        base_url = openai.base_url,
        follow_redirects = True,
    ),
)

def openai_call(messages, temperature=0.0, model="gpt-4o-2024-11-20"):
    messages =[{'role': 'system', 'content': 'You are an intelligent AI assistant.'},
                    {'role': 'user', 'content': messages}]
    fails = 0
    while True:
        if fails > 30:
            print('Failed too many times, exiting...')
            return None
        try:
            completion = client.chat.completions.create(
                model = model,
                messages = messages,
                temperature = temperature,
            )
            response = completion.choices[0].message.content
            break
        except Exception as e:
            print(f'Error: {e}')
            fails += 1
            time.sleep(fails * 0.1 + 1)
    return response

if __name__ == "__main__":
    prompt = "Who are you?"
    response = openai_call(prompt)
    
    print(response)
