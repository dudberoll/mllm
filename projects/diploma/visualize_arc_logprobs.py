import json
import os
import torch
from transformers import AutoTokenizer

# ANSI colors for visualization
def get_color(logprob):
    if logprob is None: return "\033[0m"
    # Convert logprob to probability
    prob = torch.exp(torch.tensor(float(logprob))).item()
    # Mapping probability to ANSI 256 colors (from red to green)
    if prob > 0.9: return "\033[38;5;46m" # Bright Green
    if prob > 0.7: return "\033[38;5;118m" # Light Green
    if prob > 0.5: return "\033[38;5;226m" # Yellow
    if prob > 0.3: return "\033[38;5;208m" # Orange
    return "\033[38;5;196m" # Red

def format_grid(grid):
    # Replicating the logic from arc_diagnostics.py
    # Note: the original script had a potential bug: "".join([" ".join(...)])
    # which joins rows without delimiters.
    return "".join([" ".join(map(str, row)) for row in grid])

def main():
    if not os.path.exists("arc_logprobs_results.json"):
        print("Error: arc_logprobs_results.json not found.")
        return

    # Load results
    with open("arc_logprobs_results.json", "r") as f:
        all_results = json.load(f)
    
    # Pick a task to visualize (e.g., the first one)
    res = all_results[0]
    task_id = res['task']
    logprobs = res['prompt_logprobs']
    
    # Load task data
    task_path = f"ARC-AGI/data/training/{task_id}"
    if not os.path.exists(task_path):
        # Try evaluation path if not in training
        task_path = f"ARC-AGI/data/evaluation/{task_id}"
        
    with open(task_path, 'r') as f:
        task_data = json.load(f)
    
    # Reconstruct prompt (must be EXACTLY as in arc_diagnostics.py)
    prompt = "You are an expert at solving puzzles. Here are some examples of input and output grids:"
    for i, pair in enumerate(task_data['train']):
        prompt += f"Example {i+1} Input:\n{format_grid(pair['input'])}\n"
        prompt += f"Example {i+1} Output:\n{format_grid(pair['output'])}\n\n"

    test_input = task_data['test'][0]['input']
    prompt += f"Test Input:\n{format_grid(test_input)}\nTest Output:\n"

    # Use a compatible tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    tokens = tokenizer.encode(prompt)
    
    if len(tokens) != len(logprobs):
        print(f"Warning: Token count mismatch! Tokens: {len(tokens)}, Logprobs: {len(logprobs)}")
        # This can happen if the base_model in diagnostics was different or tokenizer settings changed.
    
    print(f"\n--- Visualizing Confidence for Task: {task_id} ---")
    print("Legend: \033[38;5;46mHigh Conf (Expected)\033[0m -> \033[38;5;196mLow Conf (Surprised)\033[0m\n")

    # Print tokens with colors
    for i, token_id in enumerate(tokens):
        token_str = tokenizer.decode([token_id])
        lp = logprobs[i] if i < len(logprobs) else None
        color = get_color(lp)
        
        # Highlight grid digits
        if token_str.strip().isdigit():
            print(f"{color}\033[1m{token_str}\033[0m", end="")
        else:
            # Clean up escape sequences for display
            display_str = token_str.replace("\n", " ")
            print(f"{color}{display_str}\033[0m", end="")
    
    print("\n\n--- End of Visualization ---")

if __name__ == "__main__":
    main()
