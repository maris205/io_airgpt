import os
from openai import OpenAI

# Make sure you have stored your API Key in the environment variable ARK_API_KEY
# Initialize the OpenAI client, reading your API Key from the environment variable
client = OpenAI(
    # This is the default path; you can configure it based on your region
    base_url="https://api.intelligence.io.solutions/api/v1",
    # Get your API Key from the environment variable
    api_key="2ecf387f-42b6-48e1-a3a1-911bf014eb1c",
)

# Non-streaming:
print("----- standard request -----")
completion = client.chat.completions.create(
    # Specify the model endpoint ID
    model="moonshotai/Kimi-K2.6",
    messages=[
        {"role": "system", "content": "You are an AI assistant"},
        {"role": "user", "content": "Hello"},
    ],
)
print(completion.choices[0].message.content)

# Streaming:
print("----- streaming request -----")
stream = client.chat.completions.create(
    # Specify the model endpoint ID
    model="moonshotai/Kimi-K2.6",
    messages=[
        {"role": "system", "content": "You are an AI assistant"},
        {"role": "user", "content": "Hello"},
    ],
    # Whether to stream the response
    stream=True,
)
for chunk in stream:
    if not chunk.choices:
        continue
    print(chunk.choices[0].delta.content, end="")
print()