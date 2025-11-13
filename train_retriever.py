import torch
import wandb
import argparse
import numpy as np
from tqdm import tqdm

from pytorch_lightning import seed_everything
from datasets import DATASETS
from config import *
from model import *
from dataloader import *
from trainer import *


def train(args, export_root=None):
    seed_everything(args.seed)
    if export_root == None:
        export_root = EXPERIMENT_ROOT + '/' + args.model_code + '/' + args.dataset_code + \
            '_' + str(args.weight_decay) + '_' + str(args.bert_dropout) + '_' + str(args.bert_attn_dropout)

    train, val, test = dataloader_factory(args)
    model = LRU(args)
    trainer = LRUTrainer(args, model, train, val, test, export_root, args.use_wandb)
    trainer.train()
    trainer.test()




def predict(model, sequences, args, topk=10):
    seq_lengths = [min(len(seq), args.bert_max_len) for seq in sequences]
    max_len = max(seq_lengths)
    
    padded_seqs = []
    for seq in sequences:
        if len(seq) > max_len:
            padded = seq[-max_len:]
        else:
            padded = [0] * (max_len - len(seq)) + seq
        padded_seqs.append(padded)
    
    input_tensor = torch.tensor(padded_seqs, dtype=torch.long).to(args.device)
    
    with torch.no_grad():
        logits, _ = model(input_tensor, None)
        last_logits = logits[:, -1, :]  
        
        topk_scores, topk_indices = torch.topk(last_logits, k=topk, dim=-1)
        
        topk_scores = topk_scores.cpu().numpy()
        topk_indices = topk_indices.cpu().numpy()
        
        results = []
        for scores, indices in zip(topk_scores, topk_indices):
            mask = indices != 0
            results.append((indices[mask], scores[mask]))
            
    return results

def load_model_for_prediction(model_path, args):
    """Load a trained model for prediction."""
    dataloader_factory(args)
    model = LRU(args)
    checkpoint = torch.load(model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    return model

if __name__ == "__main__":
    set_template(args)
    if args.mode == 'train':
        print("Starting training...")
        train(args)
    elif args.mode == 'predict':
        if not args.model_path:
            raise ValueError("Please provide --model_path for prediction mode")
            
        print(f"Loading model from {args.model_path}...")
        model = load_model_for_prediction(args.model_path, args)
        print("Model loaded successfully!")
        
        print("\nExample prediction:")
        test_sequences = [
            [1, 2, 322, 4, 53, 7, 8,3,  84, 223],    # Example sequence 1
            [5, 6, 7, 8, 9],  # Example sequence 2
        ]
        results = predict(model, test_sequences, args, topk=3)
        for i, (indices, scores) in enumerate(results):
            print(f"\nSequence {i+1} predictions:")
            for idx, score in zip(indices, scores):
                print(f"  Item {idx}: {score:.4f}")

