import requests

response = requests.post(
    "http://127.0.0.1:11434/api/generate",
    json={"model": "qwen3.5:4b", "prompt": "Hello",
          "stream": False, "think": False }
)

data = response.json()
print(data["response"])
