import asyncio
import tinker
from tinker import types
from transformers import AutoTokenizer

async def main():
    service_client = tinker.ServiceClient()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    
    checkpoint_path = "tinker://1dd8de47-be86-54d3-9355-ebf80827be26:train:0/sampler_weights/000080"
    base_model = "Qwen/Qwen3-8B-Base"
    
    sampling_client = await service_client.create_sampling_client_async(
        model_path=checkpoint_path,
        base_model=base_model
    )

    task = "Solve: 3x + 10 = 25"
    prompt_text = f"<|im_start|>user\n{task}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    
    tokens = tokenizer.encode(prompt_text)
    model_input = types.ModelInput.from_ints(tokens)
    params = types.SamplingParams(max_tokens=256, temperature=0.7)
    
    try:
        response = await sampling_client.sample_async(
            prompt=model_input,
            sampling_params=params,
            num_samples=1
        )
        print(f"\nResponse Object Type: {type(response)}")
        # Пробуем найти текст
        if hasattr(response, 'text'): print(f"Text: {response.text}")
        elif hasattr(response, 'samples'): print(f"Sample 0: {response.samples[0].text}")
        else: print(f"Raw Object: {response}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
