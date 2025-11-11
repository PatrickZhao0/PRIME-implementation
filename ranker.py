
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# Load the Qwen2.5-1.5B-Instruct model
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define a simple prompt
prompt = """
You are a calulator. Your task is to calculate the result of 1+1, what is your answer? 
Answer: 
"""
# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt", padding=True)

# Get the logits for the next token
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0, -1]  # Get the logits for the last token position
    
# Get the top 5 most probable next tokens
probs = torch.nn.functional.softmax(logits, dim=-1)
topk_probs, topk_indices = torch.topk(probs, 5)

# Convert token IDs to tokens and their probabilities
topk_tokens = [tokenizer.decode([idx]) for idx in topk_indices]
topk_probs = topk_probs.tolist()

# Print the top 5 most probable next tokens and their probabilities
print("Top 5 most probable next tokens:")
for token, prob in zip(topk_tokens, topk_probs):
    # Replace newlines and other special characters for better display
    token_display = token.replace('\n', '\\n').replace('\t', '\\t')
    print(f"Token: {token_display:<20} Probability: {prob:.4f}")
