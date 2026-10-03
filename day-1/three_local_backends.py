from pathlib import Path

import ollama
import lmstudio as lms


def ollama_response(prompt):
    response = ollama.generate(model="qwen3.5:4b", prompt=prompt, think=False)
    return response["response"]


def lmstudio_response(prompt):
    model = lms.llm("qwen3.5-4b-mlx")
    return model.respond(prompt + " /no_think").content


def mlx_response(prompt):
    # Imported here (not at module top) so the ollama/lmstudio backends still
    # run on non-Apple-Silicon machines where mlx-vlm isn't installed.
    from mlx_vlm import load, generate as mlx_generate

    model_path = Path.home() / ".lmstudio/models/Qwen3.5-4B-MLX-4bit"
    model, processor = load(str(model_path))
    formatted = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    return mlx_generate(model, processor, formatted, verbose=False).text


if __name__ == "__main__":
    q = "Hello"
    print("ollama   :", ollama_response(q))
    print("lmstudio :", lmstudio_response(q))
    print("mlx      :", mlx_response(q))
