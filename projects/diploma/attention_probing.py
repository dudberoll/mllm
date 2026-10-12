import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def format_grid(grid):
    return "\n".join([" ".join(map(str, row)) for row in grid])

def format_task(task_data):
    prompt = "You are an expert at solving ARC puzzles. Here are some examples of input and output grids:\n\n"
    for i, pair in enumerate(task_data['train']):
        prompt += f"Example {i+1} Input:\n{format_grid(pair['input'])}\n"
        prompt += f"Example {i+1} Output:\n{format_grid(pair['output'])}\n\n"
    
    # We only take the first test pair for probing
    test_input = task_data['test'][0]['input']
    prompt += f"Test Input:\n{format_grid(test_input)}\nTest Output:\n"
    return prompt

def extract_attention(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # attentions is a tuple of (layer, batch, head, seq_len, seq_len)
    # We want the attention from the last token
    # shape: (num_layers, num_heads, seq_len)
    attentions = torch.stack(outputs.attentions) # [layers, batch, heads, seq, seq]
    last_token_attention = attentions[:, 0, :, -1, :] # [layers, heads, seq]
    
    return last_token_attention.cpu().numpy()

def main():
    model_id = "Qwen/Qwen2.5-0.5B"
    device = "cpu" # Default to cpu for now, or check for mps/cuda
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if device != "cpu" else torch.float32).to(device)
    model.eval()

    task_dir = "ARC-AGI/data/training"
    task_files = sorted(os.listdir(task_dir))[:5] # Process first 5 for test
    
    results = {}

    for task_file in tqdm(task_files):
        with open(os.path.join(task_dir, task_file), 'r') as f:
            task_data = json.load(f)
        
        prompt = format_task(task_data)
        attn = extract_attention(model, tokenizer, prompt, device)
        results[task_file] = attn

    np.savez("attention_results.npz", **results)
    print("Saved attention results to attention_results.npz")

if __name__ == "__main__":
    main()
