import tinker
import urllib.request

import os
from dotenv import load_dotenv

load_dotenv()

sampling_path = "tinker://efb654d6-26cf-51b4-9a55-2b902755572a:train:0/weights/final"
service_client = tinker.ServiceClient()

# sampling_client = service_client.create_sampling_client(model_path=sampling_path) 


rest_client = service_client.create_rest_client()
future = rest_client.get_weights_info_by_tinker_path(sampling_path)
response = future.result()  # or await future
print(f"Base Model: {response.base_model}, LoRA Rank: {response.lora_rank}")


tokenizer = sampling_client.get_tokenizer()
renderer = renderers.get_renderer('qwen3', tokenizer)

# 3. Создание промпта через сообщения (рекомендуемый способ)
messages = [
    {'role': 'user', 'content': 'Every morning Aya goes for a $9$-kilometer-long walk...'}
]
prompt = renderer.build_generation_prompt(messages)

# 4. Сэмплирование
stop_sequences = renderer.get_stop_sequences()
params = types.SamplingParams(
    max_tokens=1024, 
    temperature=0, # Greedy sampling для точности в математике
    stop=stop_sequences
)

future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
result = future.result()

# 5. Парсинг и вывод ответа
sampled_message, parse_success = renderer.parse_response(result.sequences[0].tokens)
print(f"Ответ модели: {sampled_message['content']}")