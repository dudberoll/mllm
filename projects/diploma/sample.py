import os
import numpy as np
import tinker
from dotenv import load_dotenv
from tinker import types

load_dotenv()
student_checkpoint = "tinker://4a1939e6-04be-5a77-9e4e-910ccff9f27e:train:0/weights/final"
service_client = tinker.ServiceClient()

# Модели
student_model = "Qwen/Qwen3-8B-Base"
# teacher_model = "openai/gpt-oss-120b"

# Клиенты
sampling_client = service_client.create_sampling_client(base_model=student_model, model_path=student_checkpoint)

tokenizer = sampling_client.get_tokenizer()

# 1. Подготовка промпта
prompt_text = "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."
prompt_tokens = tokenizer.encode(prompt_text)
prompt = types.ModelInput.from_ints(prompt_tokens)

# 2. Генерация ответа студента
params = types.SamplingParams(max_tokens=1024, temperature=0)
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
rc = service_client.create_rest_client()
future = rc.get_checkpoint_archive_url_from_tinker_path(student_checkpoint)
result = future.result()

seq = result.sequences[0]
response_tokens = seq.tokens
print(f"Student Response: {repr(tokenizer.decode(response_tokens))}")

# 3. Расчет logprobs учителя на ответ студента
# Объединение промпта и ответа для оценки контекста
# full_tokens = prompt_tokens + list(response_tokens)
# full_input = types.ModelInput.from_ints(full_tokens)

# # Получение вероятностей от модели-учителя
# teacher_logprobs_full = teacher_client.compute_logprobs(full_input).result()

# # Извлечение logprobs только для токенов ответа (смещение на длину промпта)
# response_logprobs = teacher_logprobs_full[len(prompt_tokens):]

# # 4. Анализ результатов
# print("\n--- Teacher Logprobs Analysis ---")
# print(f"Number of tokens: {len(response_logprobs)}")
# print(f"Mean Teacher Logprob: {np.mean(response_logprobs):.4f}")