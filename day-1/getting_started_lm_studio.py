import requests

response = requests.post("http://127.0.0.1:1234/v1/chat/completions",
            json={"model": "qwen3.5-4b",
            "messages": [{"role": "user", "content": "Hello"}]}
)

data = response.json()

if "choices" not in data:
    raise SystemExit(
        "LM Studio did not return a completion. Response was:\n"
        f"{data}\n\n"
        "Make sure the server is running and a model is loaded "
        "(see available models at http://127.0.0.1:1234/v1/models, "
        "or run `lms load <model>`)."
    )

print(data["choices"][0]["message"]["content"].strip())
