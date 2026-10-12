import asyncio
import tinker
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    service_client = tinker.ServiceClient()
    checkpoint_path = "tinker://4a1939e6-04be-5a77-9e4e-910ccff9f27e:train:0/sampler_weights/final"
    base_model = "Qwen/Qwen3-8B-Base"
    
    print(f"Testing access to checkpoint: {checkpoint_path}")
    try:
        # Пытаемся создать клиент. Если веса не существуют, это упадет здесь.
        client = await service_client.create_sampling_client_async(
            base_model=base_model,
            model_path=checkpoint_path
        )
        print("\n✅ SUCCESS: Checkpoint exists and is accessible!")
        print(f"Client ID: {client.sampling_session_id}")
    except Exception as e:
        print("\n❌ FAILED: Checkpoint is not accessible.")
        print(f"Error details: {e}")

if __name__ == "__main__":
    asyncio.run(main())
