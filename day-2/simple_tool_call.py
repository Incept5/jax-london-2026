import json
import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3.5:4b"
# Thinking off: ~1.5 s per question, but "Was that a good rate?" is unreliable (right in 1-2 of 5 runs;
# sometimes it converts the wrong way round). Thinking on: ~5 s per question, right 5 of 5 times, with the
# working shown (market ~1.32 vs your 1.20, about $12 short). Tested with qwen3.5:4b, 2 Oct 2026.
THINKING = False


RATES_URL = "https://api.frankfurter.dev/v1/latest"   # European Central Bank daily rates: free, no API key

# Fallback if the internet is down: units per 1 USD (ECB reference rates, 2 Oct 2026)
USD_RATES = {"USD": 1.0, "EUR": 0.89087, "GBP": 0.75753, "JPY": 157.67, "MXN": 18.335, "TRY": 49.145}


def get_rate(from_currency, to_currency):
    """Today's rate from the ECB (any of its ~30 currencies), or the built-in table if offline."""
    if from_currency == to_currency:
        return 1.0, "same currency"
    try:
        reply = requests.get(RATES_URL, params={"base": from_currency, "symbols": to_currency}, timeout=5)
        if reply.status_code == 200:
            data = reply.json()
            return data["rates"][to_currency], f"ECB, {data['date']}"
        return None, f"no ECB rate for {from_currency} to {to_currency}"   # unknown currency code
    except requests.RequestException:
        if from_currency in USD_RATES and to_currency in USD_RATES:
            return USD_RATES[to_currency] / USD_RATES[from_currency], "built-in table (offline)"
        return None, "offline, and not in the built-in table"


def convert_currency(amount, from_currency, to_currency):
    from_currency, to_currency = from_currency.upper(), to_currency.upper()
    rate, source = get_rate(from_currency, to_currency)
    if rate is None:
        return {"error": source, "result_text": f"Could not convert {from_currency} to {to_currency}: {source}"}

    result = amount * rate
    return {
        "converted_amount": round(result, 2),
        "rate": rate,
        "source": source,
        "result_text": f"{amount} {from_currency} = {result:.2f} {to_currency} ({source})"
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount from one currency to another at today's exchange rate. Use it for any question about exchange rates, including whether an exchange someone was offered was a good deal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "The amount to convert"
                    },
                    "from_currency": {
                        "type": "string",
                        "description": "The source currency as a three-letter ISO code (e.g., USD, EUR, GBP, JPY, MXN, TRY, CHF, INR)"
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "The target currency as a three-letter ISO code (e.g., USD, EUR, GBP, JPY, MXN, TRY, CHF, INR)"
                    }
                },
                "required": ["amount", "from_currency", "to_currency"]
            }
        }
    }
]


def chat_with_tools(user_message):
    print(f"USER: {user_message}\n")

    messages = [{"role": "user", "content": user_message}]

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "think": THINKING,
        }
    )

    assistant_message = response.json().get("message", {})
    tool_calls = assistant_message.get("tool_calls", [])

    if tool_calls:
        print(f"Calling {len(tool_calls)} function(s)\n")

        messages.append(assistant_message)

        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = tool_call["function"]["arguments"]

            print(f"Function: {function_name}")
            print(f"Arguments: {json.dumps(function_args)}")

            if function_name == "convert_currency":
                result = convert_currency(
                    amount=function_args["amount"],
                    from_currency=function_args["from_currency"],
                    to_currency=function_args["to_currency"]
                )

                print(f"Result: {result['result_text']}\n")

                messages.append({
                    "role": "tool",
                    "content": json.dumps(result)
                })

        final_response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "think": THINKING,
            }
        )

        final_message = final_response.json().get("message", {}).get("content", "")
        print(f"ASSISTANT: {final_message}\n")

    else:
        content = assistant_message.get("content", "")
        print(f"ASSISTANT: {content}\n")


if __name__ == "__main__":
    print(f"Model: {MODEL}\n")
    print("-" * 60)

    chat_with_tools("What is EUR 50 in USD?")
    print("-" * 60)

    chat_with_tools("Convert 100 USD to GBP")
    print("-" * 60)

    chat_with_tools("How many US dollars is 500 Mexican pesos?")
    print("-" * 60)

    chat_with_tools("How many Turkish lira do I get for 200 dollars?")
    print("-" * 60)

    chat_with_tools("What are 1,000 Indian rupees in Swiss francs?")   # any pair: the rate comes live from the ECB
    print("-" * 60)

    # Needs the tool AND some reasoning: £100 is worth about $132, so $120 was a poor rate. Run it a few times.
    # Then try "I was given $120 for £100, was this good?": small models often convert the wrong way round.
    chat_with_tools("I changed £100 into dollars and got $120. Was that a good rate?")
    print("-" * 60)

    chat_with_tools("What is the capital of France?")
    print("-" * 60)
