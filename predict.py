import os
import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from model import LRU
from config import *
from dataloader import *
import numpy as np

def load_model(model_path, args):
    """Load the trained model from checkpoint."""
    model = LRU(args)
    checkpoint = torch.load(model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    return model

def preprocess_sequences(sequences, max_len, pad_token=0):
    """
    Preprocess input sequences to match model's expected format.
    
    Args:
        sequences: List of lists, where each inner list is a sequence of item indices
        max_len: Maximum sequence length
        pad_token: Token used for padding shorter sequences
        
    Returns:
        Tensor of shape (batch_size, max_len) containing padded sequences
    """
    processed = []
    for seq in sequences:
        # Truncate if sequence is longer than max_len
        if len(seq) > max_len:
            seq = seq[-max_len:]
        # Pad if sequence is shorter than max_len
        if len(seq) < max_len:
            seq = [pad_token] * (max_len - len(seq)) + seq
        processed.append(seq)
    return torch.tensor(processed, dtype=torch.long)

def predict_next_item(model, sequences, args, topk=10):
    """
    Predict the next item for each sequence in the input.
    
    Args:
        model: Trained LRU model
        sequences: List of lists, where each inner list is a sequence of item indices
        args: Model arguments
        topk: Number of top predictions to return
        
    Returns:
        List of tuples containing (topk_indices, topk_scores) for each sequence
    """
    # Preprocess sequences
    input_tensor = preprocess_sequences(sequences, args.bert_max_len)
    input_tensor = input_tensor.to(args.device)
    
    # Get model predictions
    with torch.no_grad():
        logits = model(input_tensor, None)
        # Get the last prediction for each sequence
        last_logits = logits[:, -1, :]  # shape: (batch_size, num_items + 1)
        
        # Get top-k predictions
        topk_scores, topk_indices = torch.topk(last_logits, k=topk, dim=-1)
        
        # Convert to numpy for easier handling
        topk_scores = topk_scores.cpu().numpy()
        topk_indices = topk_indices.cpu().numpy()
        
        # Filter out padding token (0) if present in top-k
        results = []
        for scores, indices in zip(topk_scores, topk_indices):
            # Filter out padding token (0) and adjust indices
            mask = indices != 0
            results.append((indices[mask], scores[mask]))
            
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the trained model checkpoint')
    parser.add_argument('--dataset_code', type=str, default='beauty', 
                       help='Dataset code (e.g., beauty, clothing)')
    parser.add_argument('--max_len', type=int, default=50, 
                       help='Maximum sequence length')
    parser.add_argument('--topk', type=int, default=10, 
                       help='Number of top predictions to return')
    args = parser.parse_args()
    
    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and dataset configuration
    dataset_config = {'beauty': beauty, 'clothing': clothing, 'sports': sports, 
                     'toys': toys, 'yelp': yelp}
    
    if args.dataset_code not in dataset_config:
        raise ValueError(f"Dataset {args.dataset_code} not supported")
    
    # Update args with dataset config
    dataset_args = dataset_config[args.dataset_code]
    for k, v in dataset_args.items():
        setattr(args, k, v)
    
    # Set model-specific arguments
    set_template(args)
    
    # Load model
    model = load_model(args.model_path, args)
    
    # Example usage
    example_sequences = [
        [1, 2, 3, 4],  # Example sequence 1
        [5, 6, 7, 8, 9],  # Example sequence 2
    ]
    
    print("Example predictions:")
    predictions = predict_next_item(model, example_sequences, args, topk=args.topk)
    
    for i, (indices, scores) in enumerate(predictions):
        print(f"\nSequence {i+1}:")
        print(f"Input sequence: {example_sequences[i]}")
        print("Top predictions:")
        for idx, score in zip(indices, scores):
            print(f"  Item {idx}: {score:.4f}")

if __name__ == "__main__":
    main()
