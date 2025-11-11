from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch
import numpy as np

# 1. Load the model and tokenizer
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
    device_map="auto"  # This will use GPU if available
)

# 2. Prepare your training data
# Example: Simple arithmetic dataset
training_examples = [
    "What is 2+2? The answer is 4.",
    "Calculate 10-5: The result is 5.",
    "If you add 3 and 4, you get 7.",
    "The sum of 1 and 1 is 2.",
    "8 divided by 2 equals 4.",
    # Add more examples...
]

# Tokenize the training data
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,  # Adjust based on your needs
        return_tensors="pt"
    )

# Create a dataset
dataset = Dataset.from_dict({"text": training_examples})
dataset = dataset.map(tokenize_function, batched=True)

# 3. Set up training arguments
training_args = TrainingArguments(
    output_dir="./qwen-finetuned",
    num_train_epochs=3,  # Adjust as needed
    per_device_train_batch_size=1,  # Reduce if you run into memory issues
    gradient_accumulation_steps=4,  # Simulate larger batch size
    save_steps=100,
    save_total_limit=2,
    learning_rate=2e-5,  # Learning rate for fine-tuning
    weight_decay=0.01,
    fp16=True,  # Use mixed precision training
    logging_dir='./logs',
    logging_steps=10,
    report_to="none",  # Change to "wandb" if you want to use Weights & Biases
    optim="adamw_torch",
)

# 4. Create a data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal language modeling
)

# 5. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# 6. Start fine-tuning
print("Starting fine-tuning...")
trainer.train()

# 7. Save the fine-tuned model
output_dir = "./qwen-finetuned-arithmetic"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

# Example of how to use the fine-tuned model
def generate_response(prompt, model, tokenizer, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the fine-tuned model
test_prompt = "What is 5+5?"
response = generate_response(test_prompt, model, tokenizer)
print(f"\nTest prompt: {test_prompt}")
print(f"Model response: {response}")
