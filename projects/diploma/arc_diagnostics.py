import json
import os
import tinker
from tinker import types
from tqdm import tqdm

def format_grid(grid):
    return "".join([" ".join(map(str, row)) for row in grid])

def format_task(task_data):
    prompt = "You are an expert at solving puzzles. Here are some examples of input and output grids:"
    for i, pair in enumerate(task_data['train']):
        prompt += f"Example {i+1} Input:\n{format_grid(pair['input'])}\n"
        prompt += f"Example {i+1} Output:\n{format_grid(pair['output'])}\n\n"

    test_input = task_data['test'][0]['input']
    prompt += f"Test Input:\n{format_grid(test_input)}\nTest Output:\n"
    return prompt

def main():
    # 1. Setup Clients
    # Note: TINKER_API_KEY must be set in environment
    service_client = tinker.ServiceClient()
    model_name = "Qwen/Qwen3-8B-Base"
    
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    tokenizer = sampling_client.get_tokenizer()

    # 2. Process 50 ARC Tasks for logprobs
    task_dir = "ARC-AGI/data/training"
    task_files = sorted([f for f in os.listdir(task_dir) if f.endswith('.json')])[:50]
    
    all_results = []
    print(f"Processing {len(task_files)} tasks for Attention Probing (logprobs)...")
    
    for task_file in tqdm(task_files):
        with open(os.path.join(task_dir, task_file), 'r') as f:
            task_data = json.load(f)
        
        prompt_text = format_task(task_data)
        tokens = tokenizer.encode(prompt_text)
        
        # We use a small max_tokens since we only need the prompt logprobs for diagnosis
        future = sampling_client.sample(
            prompt=types.ModelInput.from_ints(tokens),
            sampling_params=types.SamplingParams(max_tokens=1, temperature=0.0),
            num_samples=1,
            include_prompt_logprobs=True,
            topk_prompt_logprobs=5
        )
        result = future.result()
        all_results.append({
            "task": task_file,
            "prompt_logprobs": result.prompt_logprobs
        })

    # Save results locally
    with open("arc_logprobs_results.json", "w") as f:
        json.dump(all_results, f)
    print("Saved logprobs to arc_logprobs_results.json")

    # 3. Export for Geometric Head Search
    print("Initializing LoRA Training Client for weight export...")
    training_client = service_client.create_lora_training_client(
        base_model=model_name, rank=32
    )
    
    print("Saving weights for sampler...")
    save_future = training_client.save_weights_for_sampler("arc_probe_weights")
    checkpoint_state = save_future.result()
    checkpoint_path = checkpoint_state.path
    print(f"Weights saved at: {checkpoint_path}")

    # 4. Get Download URL
    print("Generating download URL via REST client...")
    rest_client = service_client.create_rest_client()
    url_future = rest_client.get_checkpoint_archive_url_from_tinker_path(checkpoint_path)
    download_url = url_future.result()
    
    print(f"SUCCESS!")
    print(f"Download URL for weights: {download_url}")
    print("You can now download the archive and use the merge script:")
    print(f"python -m tinker_cookbook.scripts.merge_tinker_adapter_to_hf_model --base_model {model_name} --adapter_path ./arc_probe_checkpoint --output_path ./merged_model")

if __name__ == "__main__":
    main()
