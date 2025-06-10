import json
import re
from typing import List, Dict, Any
import pandas as pd

def convert_glaive_to_qwen3(input_file: str, output_file: str) -> None:
“””
Convert Glaive function calling dataset to Qwen3 format.

```
Args:
    input_file: Path to input CSV/JSON file with 'chat' column
    output_file: Path to output JSONL file
"""
# Load dataset
if input_file.endswith('.csv'):
    df = pd.read_csv(input_file)
else:
    df = pd.read_json(input_file, lines=True)

converted_conversations = []

for idx, row in df.iterrows():
    chat_text = row['chat']
    messages = parse_conversation(chat_text)
    
    if messages:
        conversation = {
            "messages": messages,
            "conversation_id": f"glaive_{idx}"
        }
        converted_conversations.append(conversation)

# Save as JSONL
with open(output_file, 'w', encoding='utf-8') as f:
    for conv in converted_conversations:
        f.write(json.dumps(conv, ensure_ascii=False) + '\n')

print(f"Converted {len(converted_conversations)} conversations to {output_file}")
```

def parse_conversation(chat_text: str) -> List[Dict[str, Any]]:
“””
Parse a single conversation from Glaive format to Qwen3 messages.

```
Format:
USER: question
ASSISTANT: <functioncall> {...} <|endoftext|>
FUNCTION RESPONSE: {...}
ASSISTANT: response <|endoftext|>
"""
messages = []

# Split by roles - more precise regex
parts = re.split(r'\n?\s*(USER:|ASSISTANT:|FUNCTION RESPONSE:)\s*', chat_text)
parts = [p.strip() for p in parts if p.strip()]

i = 0
while i < len(parts):
    if parts[i] == 'USER:' and i + 1 < len(parts):
        # User message
        content = clean_content(parts[i + 1])
        messages.append({
            "role": "user", 
            "content": content
        })
        i += 2
        
    elif parts[i] == 'ASSISTANT:' and i + 1 < len(parts):
        # Assistant message - check for function call
        content = clean_content(parts[i + 1])
        
        # Extract function call
        func_call = extract_function_call(content)
        if func_call:
            # Assistant with tool call
            msg = {
                "role": "assistant",
                "tool_calls": [func_call]
            }
            # Add remaining content if any
            remaining = remove_function_call(content)
            if remaining:
                msg["content"] = remaining
            messages.append(msg)
            
            # Look for function response
            if i + 3 < len(parts) and parts[i + 2] == 'FUNCTION RESPONSE:':
                response_content = clean_content(parts[i + 3])
                messages.append({
                    "role": "tool",
                    "tool_call_id": func_call["id"],
                    "content": response_content
                })
                i += 4
                
                # Check for follow-up assistant message
                if i + 1 < len(parts) and parts[i] == 'ASSISTANT:':
                    follow_up = clean_content(parts[i + 1])
                    messages.append({
                        "role": "assistant",
                        "content": follow_up
                    })
                    i += 2
            else:
                i += 2
        else:
            # Regular assistant message
            messages.append({
                "role": "assistant",
                "content": content
            })
            i += 2
    else:
        i += 1

return messages
```

def clean_content(text: str) -> str:
“”“Remove markers and clean text content.”””
# Remove <|endoftext|> markers
text = re.sub(r’<|endoftext|>’, ‘’, text)
# Remove extra whitespace
text = re.sub(r’\s+’, ’ ’, text)
return text.strip()

def extract_function_call(text: str) -> Dict[str, Any] | None:
“”“Extract function call from text.”””
# Match <functioncall> {…} </functioncall> or <functioncall> {…} <|endoftext|>
pattern = r’<functioncall>\s*({[^}]*})\s*(?:</functioncall>|<|endoftext|>)’
match = re.search(pattern, text, re.DOTALL)

```
if match:
    try:
        func_data = json.loads(match.group(1))
        return {
            "id": f"call_{abs(hash(match.group(1))) % 10000:04d}",
            "type": "function",
            "function": {
                "name": func_data["name"],
                "arguments": func_data["arguments"]
            }
        }
    except (json.JSONDecodeError, KeyError):
        return None
return None
```

def remove_function_call(text: str) -> str:
“”“Remove function call from text and return remaining content.”””
pattern = r’<functioncall>.*?(?:</functioncall>|<|endoftext|>)’
cleaned = re.sub(pattern, ‘’, text, flags=re.DOTALL)
return clean_content(cleaned)

# Test with sample data

def test_converter():
“”“Test the converter with the provided sample.”””
sample = ‘’‘USER: Can you tell me the latest news headlines for the United States?   ASSISTANT: <functioncall> {“name”: “get_news_headlines”, “arguments”: ‘{“country”: “United States”}’} <|endoftext|>   FUNCTION RESPONSE: {“headlines”: [“Biden announces new vaccine mandates”, “Hurricane Ida devastates Louisiana”, “Apple unveils new iPhone”, “NASA’s Perseverance rover collects first Mars rock sample”]}   ASSISTANT: Here are the latest news headlines for the United States: 1. Biden announces new vaccine mandates 2. Hurricane Ida devastates Louisiana 3. Apple unveils new iPhone 4. NASA’s Perseverance rover collects first Mars rock sample <|endoftext|>   USER: That’s interesting. What about the news in France?   ASSISTANT: <functioncall> {“name”: “get_news_headlines”, “arguments”: ‘{“country”: “France”}’} <|endoftext|>   FUNCTION RESPONSE: {“headlines”: [“France recalls ambassadors to US and Australia”, “French election: Macron’s party braces for tough fight”, “Louvre Museum to undergo major overhaul”, “France to offer free birth control to all women under 25”]}   ASSISTANT: Here are the latest news headlines for France: 1. France recalls ambassadors to US and Australia 2. French election: Macron’s party braces for tough fight 3. Louvre Museum to undergo major overhaul 4. France to offer free birth control to all women under 25 <|endoftext|>’’’

```
messages = parse_conversation(sample)

print("Converted Messages:")
for i, msg in enumerate(messages):
    print(f"{i+1}. Role: {msg['role']}")
    if 'content' in msg:
        print(f"   Content: {msg['content'][:100]}...")
    if 'tool_calls' in msg:
        print(f"   Tool: {msg['tool_calls'][0]['function']['name']}")
    print()
```

# Usage example

if **name** == “**main**”:
# Test the converter
test_converter()

```
# Convert actual dataset
# convert_glaive_to_qwen3('glaive-function-calling-v2.csv', 'qwen3_dataset.jsonl')
```
