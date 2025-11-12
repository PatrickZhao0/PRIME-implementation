import torch
import argparse
from model import LRU

def load_model(args, model_path):
    model = LRU(args)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint[STATE_DICT_KEY])
    model.eval()
    return model

def predict(model, seq, k=10):
    if not isinstance(seq, torch.Tensor):
        seq = torch.tensor(seq, dtype=torch.long).unsqueeze(0)  
    
    with torch.no_grad():
        scores = model(seq)[0][:, -1, :]  
        
        for i in range(seq.size(1)):
            scores[torch.arange(scores.size(0)), seq[:, i]] = -1e9
        scores[:, 0] = -1e9  
        
        topk_scores, topk_items = torch.topk(scores, k=k, dim=-1)
        
    return topk_items.squeeze().tolist(), topk_scores.squeeze().tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the trained model checkpoint')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of recommendations to generate (default: 10)')

    
    args = parser.parse_args()
    
    model = load_model(args, f'experiments/lru/{args.model_path}_0.01_0.2_0.2/models/best_acc_model.pth')
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    input_sequence = [1, 2, 3, 4, 5]  
    
    topk_items, topk_scores = predict(model, input_sequence, k=args.k)
    
    print(f"Input sequence: {input_sequence}")
    print(f"Top-{args.k} recommended items: {topk_items}")
    print(f"Scores: {topk_scores}")

if __name__ == "__main__":
    main()
