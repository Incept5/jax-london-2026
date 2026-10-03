import base64
import json
from pathlib import Path
import requests

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def main():
    payload = {
        "model": "qwen3.5:4b",
        "messages": [{
                "role": "system",
                "content": "You are a helpful visual assistant, give detailed and accurate answers.",
            },{
                "role": "user", "content": "List the ice cream flavours shown on the menu in this image.",
                "images": [encode_image_to_base64("data/IMG_3319.jpeg")], },],
        "stream": False,
        "think": False,
        "options": { "temperature": 0.7, "num_predict": 1000, },
    }

    response = requests.post(
        "http://127.0.0.1:11434/api/chat", headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=120,
    )
    response.raise_for_status()
    print(response.json()["message"]["content"])

if __name__ == "__main__":
    main()
