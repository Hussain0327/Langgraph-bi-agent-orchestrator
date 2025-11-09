#!/usr/bin/env python3

from src.config import Config
from openai import OpenAI
import json

client = OpenAI(api_key=Config.OPENAI_API_KEY)
response = client.responses.create(
    model=Config.OPENAI_MODEL,
    input='What is 2+2?',
    max_output_tokens=20
)

print("=== RESPONSE ATTRIBUTES ===")
print(f"output_text: '{response.output_text}'")
print(f"status: {response.status}")
print(f"error: {response.error}")

print("\n=== OUTPUT ARRAY ===")
print(f"output length: {len(response.output)}")

for i, item in enumerate(response.output):
    print(f"\nItem {i}:")
    print(f"  type: {item.type}")
    print(f"  id: {item.id}")

    if item.type == 'message':
        print(f"  role: {item.role}")
        print(f"  status: {item.status}")
        print(f"  content: {item.content}")
        if item.content:
            for c in item.content:
                if hasattr(c, 'text'):
                    print(f"    text: '{c.text}'")

    if item.type == 'reasoning':
        print(f"  summary: {item.summary}")
        print(f"  content: {item.content}")
        print(f"  status: {item.status}")

print("\n=== FULL RESPONSE ===")
print(response)
print("\n=== RESPONSE AS DICT ===")
print(json.dumps(response.to_dict(), indent=2))
