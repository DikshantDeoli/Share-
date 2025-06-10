import json
import re
from typing import List, Dict, Any, Optional
import pandas as pd

class GlaiveToQwen3Converter:
“””
Converts Glaive function calling dataset format to Qwen3 style messages format.

```
Glaive format uses:
- USER: / ASSISTANT: markers
- <functioncall> tags for tool calls
- FUNCTION RESPONSE: for tool responses
- <|endoftext|> as separators

Qwen3 format uses:
- Standard chat messages with roles
- Structured tool calls and responses
"""

def __init__(self):
    self.conversation_pattern = re.compile(
        r'(USER:|ASSISTANT:|FUNCTION RESPONSE:)\s*(.*?)(?=(?:USER:|ASSISTANT:|FUNCTION RESPONSE:|<\|endoftext\|>|$))',
        re.DOTALL | re.MULTILINE
    )
    self.function_call_pattern = re.compile(
        r'<functioncall>\s*({.*?})\s*</functioncall>',
        re.DOTALL
    )

def extract_function_call(self, text: str) -> tuple[Optional[Dict], str]:
    """
    Extract function call from assistant message text.
    Returns (function_call_dict, remaining_text)
    """
    match = self.function_call_pattern.search(text)
    if match:
        try:
            func_call_str = match.group(1).strip()
            func_call = json.loads(func_call_str)
            
            # Remove the function call from text
            remaining_text = self.function_call_pattern.sub('', text).strip()
            
            # Convert to Qwen3 tool call format
            tool_call = {
                "id": f"call_{hash(func_call_str) % 10000:04d}",
                "type": "function",
                "function": {
                    "name": func_call["name"],
                    "arguments": func_call["arguments"]
                }
            }
            return tool_call, remaining_text
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing function call: {e}")
            return None, text
    return None, text

def parse_function_response(self, response_text: str) -> Dict[str, Any]:
    """
    Parse function response text into structured format.
    """
    response_text = response_text.strip()
    try:
        # Try to parse as JSON
        return json.loads(response_text)
    except json.JSONDecodeError:
        # If not valid JSON, return as string content
        return {"content": response_text}

def split_conversation_parts(self, chat_text: str) -> List[tuple[str, str]]:
    """
    Split the chat text into (role, content) pairs.
    """
    parts = []
    matches = self.conversation_pattern.findall(chat_text)
    
    for role_marker, content in matches:
        role_marker = role_marker.strip().rstrip(':')
        content = content.strip()
        
        # Skip empty content
        if not content or content == '<|endoftext|>':
            continue
        
        # Clean up content - remove trailing <|endoftext|>
        content = re.sub(r'<\|endoftext\|>\s*$', '', content).strip()
        
        if content:  # Only add non-empty content
            parts.append((role_marker, content))
    
    return parts

def convert_single_conversation(self, chat_text: str) -> List[Dict[str, Any]]:
    """
    Convert a single conversation from Glaive format to Qwen3 format.
    """
    messages = []
    parts = self.split_conversation_parts(chat_text)
    
    i = 0
    while i < len(parts):
        role, content = parts[i]
        
        if role == "USER":
            messages.append({
                "role": "user",
                "content": content
            })
            i += 1
            
        elif role == "ASSISTANT":
            # Check if this assistant message contains a function call
            tool_call, remaining_content = self.extract_function_call(content)
            
            if tool_call:
                # Assistant message with tool call
                assistant_msg = {
                    "role": "assistant",
                    "tool_calls": [tool_call]
                }
                
                # Add content if there's any remaining text
                if remaining_content:
                    assistant_msg["content"] = remaining_content
                
                messages.append(assistant_msg)
                
                # Look for the corresponding function response
                if i + 1 < len(parts) and parts[i + 1][0] == "FUNCTION RESPONSE":
                    _, func_response_text = parts[i + 1]
                    func_response = self.parse_function_response(func_response_text)
                    
                    # Add tool response message
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(func_response) if isinstance(func_response, dict) else str(func_response)
                    })
                    
                    i += 2  # Skip the function response part
                else:
                    i += 1
                    
                # Check if there's a follow-up assistant message after the function call
                if i < len(parts) and parts[i][0] == "ASSISTANT":
                    _, follow_up_content = parts[i]
                    messages.append({
                        "role": "assistant",
                        "content": follow_up_content
                    })
                    i += 1
                    
            else:
                # Regular assistant message without tool call
                messages.append({
                    "role": "assistant",
                    "content": content
                })
                i += 1
                
        elif role == "FUNCTION RESPONSE":
            # This should be handled as part of assistant message processing
            # If we encounter it here, it means there was no preceding function call
            print(f"Warning: Found orphaned function response: {content[:100]}...")
            i += 1
        else:
            print(f"Warning: Unknown role marker: {role}")
            i += 1
    
    return messages

def convert_dataset(self, input_data, chat_column: str = 'chat') -> List[Dict[str, Any]]:
    """
    Convert entire dataset from Glaive format to Qwen3 format.
    
    Args:
        input_data: Can be a pandas DataFrame, list of dicts, or path to JSON/CSV file
        chat_column: Name of the column containing chat data
        
    Returns:
        List of conversations in Qwen3 format
    """
    # Handle different input types
    if isinstance(input_data, str):
        # Assume it's a file path
        if input_data.endswith('.csv'):
            df = pd.read_csv(input_data)
        elif input_data.endswith('.json') or input_data.endswith('.jsonl'):
            df = pd.read_json(input_data, lines=input_data.endswith('.jsonl'))
        else:
            raise ValueError("Unsupported file format. Use CSV, JSON, or JSONL.")
    elif isinstance(input_data, pd.DataFrame):
        df = input_data
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    else:
        raise ValueError("Input data must be a file path, DataFrame, or list of dicts")
    
    if chat_column not in df.columns:
        raise ValueError(f"Column '{chat_column}' not found in dataset")
    
    converted_conversations = []
    
    for idx, row in df.iterrows():
        chat_text = row[chat_column]
        
        if pd.isna(chat_text) or not chat_text.strip():
            print(f"Warning: Empty or NaN chat text at row {idx}")
            continue
            
        try:
            messages = self.convert_single_conversation(chat_text)
            
            if messages:  # Only add non-empty conversations
                conversation = {
                    "messages": messages,
                    "source": "glaive-function-calling-v2",
                    "conversation_id": f"glaive_{idx}"
                }
                
                # Add any additional columns from the original dataset
                for col in df.columns:
                    if col != chat_column:
                        conversation[f"original_{col}"] = row[col]
                
                converted_conversations.append(conversation)
                
        except Exception as e:
            print(f"Error converting conversation at row {idx}: {e}")
            print(f"Chat text preview: {str(chat_text)[:200]}...")
            continue
    
    return converted_conversations

def save_converted_dataset(self, converted_data: List[Dict], output_path: str, format: str = 'jsonl'):
    """
    Save converted dataset to file.
    
    Args:
        converted_data: List of converted conversations
        output_path: Output file path
        format: Output format ('jsonl', 'json', or 'csv')
    """
    if format == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for conversation in converted_data:
                f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
                
    elif format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
            
    elif format == 'csv':
        # For CSV, we'll flatten the structure somewhat
        flattened_data = []
        for conv in converted_data:
            flattened_data.append({
                'conversation_id': conv.get('conversation_id', ''),
                'source': conv.get('source', ''),
                'messages_json': json.dumps(conv['messages'], ensure_ascii=False),
                'num_messages': len(conv['messages'])
            })
        pd.DataFrame(flattened_data).to_csv(output_path, index=False)
    else:
        raise ValueError("Format must be 'jsonl', 'json', or 'csv'")
```

# Example usage and testing

def test_converter():
“”“Test the converter with the provided sample.”””

```
sample_chat = '''USER: Can you tell me the latest news headlines for the United States?   ASSISTANT: <functioncall> {"name": "get_news_headlines", "arguments": '{"country": "United States"}'} <|endoftext|>   FUNCTION RESPONSE: {"headlines": ["Biden announces new vaccine mandates", "Hurricane Ida devastates Louisiana", "Apple unveils new iPhone", "NASA's Perseverance rover collects first Mars rock sample"]}   ASSISTANT: Here are the latest news headlines for the United States: 1. Biden announces new vaccine mandates 2. Hurricane Ida devastates Louisiana 3. Apple unveils new iPhone 4. NASA's Perseverance rover collects first Mars rock sample <|endoftext|>   USER: That's interesting. What about the news in France?   ASSISTANT: <functioncall> {"name": "get_news_headlines", "arguments": '{"country": "France"}'} <|endoftext|>   FUNCTION RESPONSE: {"headlines": ["France recalls ambassadors to US and Australia", "French election: Macron's party braces for tough fight", "Louvre Museum to undergo major overhaul", "France to offer free birth control to all women under 25"]}   ASSISTANT: Here are the latest news headlines for France: 1. France recalls ambassadors to US and Australia 2. French election: Macron's party braces for tough fight 3. Louvre Museum to undergo major overhaul 4. France to offer free birth control to all women under 25 <|endoftext|>'''

converter = GlaiveToQwen3Converter()

# Test single conversation conversion
messages = converter.convert_single_conversation(sample_chat)

print("Converted messages:")
print(json.dumps(messages, indent=2, ensure_ascii=False))

print(f"\nTotal messages: {len(messages)}")
print("Message types:", [msg.get('role') for msg in messages])
```

# Usage example for full dataset conversion

def convert_glaive_dataset(input_file: str, output_file: str):
“””
Complete example of converting a Glaive dataset file.
“””
converter = GlaiveToQwen3Converter()

```
# Convert the dataset
print("Converting dataset...")
converted_data = converter.convert_dataset(input_file, chat_column='chat')

print(f"Converted {len(converted_data)} conversations")

# Save the converted dataset
converter.save_converted_dataset(converted_data, output_file, format='jsonl')
print(f"Saved converted dataset to {output_file}")

# Print sample statistics
if converted_data:
    sample_conv = converted_data[0]
    print(f"\nSample conversation structure:")
    print(f"- Conversation ID: {sample_conv.get('conversation_id')}")
    print(f"- Number of messages: {len(sample_conv['messages'])}")
    print(f"- Message roles: {[msg.get('role') for msg in sample_conv['messages']]}")
```

if **name** == “**main**”:
# Run the test
test_converter()

```
# Example of how to use for full dataset conversion:
# convert_glaive_dataset('glaive-function-calling-v2.jsonl', 'qwen3_converted_dataset.jsonl')
```
