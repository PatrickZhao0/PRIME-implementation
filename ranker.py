
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# Load the Qwen2.5-1.5B-Instruct model
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define a simple prompt
prompt = "You are a recommendation ranker. Your task is to rank multiple "
prompt += "candidate items based on their relevance to the user's preferences. "
prompt += "Your input is a JSON object with the structure: "
prompt += "{ 'user': {'history': [item1, item2, ...], 'context': 'user context'}, "
prompt += "'candidates': [item1, item2, ...] }. "
prompt += "Your output should be a JSON object with the structure: "
prompt += "{ 'ranked_items': [item1, item2, ...] }. "
prompt += "For example, if you are given the user history [item1, item2] and "
prompt += "the candidates [item2, item3, item1], your output should be "
prompt += "{ 'ranked_items': [item2, item1, item3] }. "

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt", padding=True)

# Generate the next tokens
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

# Print the generated response
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
