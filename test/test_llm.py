from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  extra_body={},
  model="deepseek/deepseek-chat-v3.1:free",
  messages=[
              {
                "role": "user",
                "content": "Salut comment tu vas ?"
              }
            ]
)
print(completion.choices[0].message.content)