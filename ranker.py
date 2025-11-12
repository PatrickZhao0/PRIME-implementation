
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

browsing_history = json.dumps({
    "title": "Hand Cream",
    "category": "Moisturizers",
    "image": "<image>",
    "price": 10.99
})
candidate_items = json.dumps([
    {
       "index": "A",
       "title": "Skin Care Bundle", 
       "price": 29.99,
       "category": "Moisturizers",
       "image": "<image>"
    },
    {
        "index": "B",
        "title": "5 in 1 Cream",
        "category": "Moisturizers",
        "image": "<image>",
        "price": 26.99
    }
])
messages = [
    {"role": "system", "content": (
        "You are a helpful assistant."
    )},
    {"role": "user", "content": (
        "Based on the user's browsing history and a collection of candidate"
        "items (both provided in JSON format), recommend the most suitable"
        "item by specifying its index letter. (Answer letter only, no other text)"
        "\n\n"
        f"Browsing history: {browsing_history}"
        "\n\n"
        f"Candidate items: {candidate_items}"
    )},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=False, output_attentions=False)
    logits = outputs.logits[0, -1]
probs = torch.nn.functional.softmax(logits, dim=-1)
topk_probs, topk_indices = torch.topk(probs, 5)


topk_tokens = [tokenizer.decode([idx]) for idx in topk_indices]
topk_probs = topk_probs.tolist()

print("\nPrompt:")
print(prompt)
print("\nTop 5 most probable next tokens:")
for token, prob in zip(topk_tokens, topk_probs):
    token_display = token.replace('\n', '\\n').replace('\t', '\\t')
    print(f"Token: {token_display:<20} Probability: {prob:.4f}")
