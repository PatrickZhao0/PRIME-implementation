import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import pickle
from model import LRU
import argparse
from dataloader.sas import SASDataloader
from datasets.beauty import BeautyDataset

def load_model(model_path, args):
    model = LRU(args)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def preprocess_sequence(sequence, max_len, num_items):
    """Preprocess the input sequence to match model's expected format."""
    # Convert to list if not already
    if isinstance(sequence, str):
        sequence = [int(x) for x in sequence.split(',')]
    
    # Truncate sequence if it's longer than max_len
    if len(sequence) > max_len:
        sequence = sequence[-max_len:]
    
    # Pad sequence if it's shorter than max_len
    if len(sequence) < max_len:
        padding = [0] * (max_len - len(sequence))
        sequence = padding + sequence
    
    # Convert to tensor
    sequence = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    return sequence

def predict_next_items(model, sequence, args, top_k=10):
    
    seq_tensor = preprocess_sequence(sequence, args.bert_max_len, args.num_items)
    
    with torch.no_grad():
        logits = model(seq_tensor, labels=None)
        scores = F.softmax(logits, dim=-1)  # Convert logits to probabilities
    top_scores, top_indices = torch.topk(scores[0, -1, :], k=top_k)
    
    top_items = top_indices.tolist()
    top_scores = top_scores.tolist()
    
    return list(zip(top_items, top_scores))

def load_dataset_and_model(model_path, dataset_name='beauty'):
    dataset = BeautyDataset()
    
    dataloader = SASDataloader(args, dataset)
    
    args.num_items = dataloader.item_count
    args.bert_max_len = getattr(args, 'bert_max_len', 100)  # Default max length
    
    model = load_model(model_path, args)
    
    return model, dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the trained model checkpoint')
    parser.add_argument('--sequence', type=str, required=True,
                       help='Comma-separated list of item IDs representing user history')
    parser.add_argument('--bert_max_len', type=int, default=100,
                       help='Maximum sequence length')
    args = parser.parse_args()
    
    model, dataloader = load_dataset_and_model(args.model_path)
    
    sequence = [int(x) for x in args.sequence.split(',')]
    
    predictions = predict_next_items(model, sequence, args, top_k=5)
    
    print("-" * 50)
    print("{:<10} {:<15} {}".format("Rank", "Item ID", "Score"))
    print("-" * 50)
    for i, (item_id, score) in enumerate(predictions, 1):
        print("{:<10} {:<15} {:.4f}".format(i, item_id, score))

if __name__ == "__main__":
    main()