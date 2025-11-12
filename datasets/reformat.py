#!/usr/bin/env python3
import argparse

def expand_sequences(input_path):
    with open(f"data/{input_path}/sequential_data.txt", "r") as fin, open(f"data/{input_path}/sequential_data_reformatted.txt", "w") as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 2:
                continue 
            user = parts[0]
            items = parts[1:]
            for item in items:
                fout.write(f"{user} {item}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand per-user sequences into (user, item) pairs.")
    parser.add_argument("--input", required=True, help="Path to sequential_data.txt")
    args = parser.parse_args()

    expand_sequences(args.input)
    print(f"Expanded pairs written to data/{args.input}/sequential_data_reformatted.txt")