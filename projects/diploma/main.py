import os
import numpy as np
from dotenv import load_dotenv
import tinker
from tinker import types

load_dotenv()

service_client = tinker.ServiceClient()

# Конфигурация моделей
# Дистиллированный чекпоинт (Step 80)
student_checkpoint = "tinker://1dd8de47-be86-54d3-9355-ebf80827be26:train:0/sampler_weights/000080"
student_base = "Qwen/Qwen3-8B-Base"
teacher_model = "Qwen/Qwen3-30B-A3B"

print(f"Loading Distilled Student Model: {student_checkpoint}")
sampling_client = service_client.create_sampling_client(
    base_model=student_base,
    model_path=student_checkpoint
)

print(f"Loading Teacher Model: {teacher_model}")
teacher_client = service_client.create_sampling_client(base_model=teacher_model)

tokenizer = sampling_client.get_tokenizer()

# Задача про Aya
problem = (
    "Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop afterwards. "
    "When she walks at a constant speed of s kilometers per hour, the walk takes her 4 hours, "
    "including t minutes spent in the coffee shop. When she walks s+2 kilometers per hour, "
    "the walk takes her 2 hours and 24 minutes, including t minutes spent in the coffee shop. "
    "Suppose Aya walks at s+1/2 kilometers per hour. Find the number of minutes the walk takes her, "
    "including the t minutes spent in the coffee shop."
)

prompt_text = f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n<think>\n"

prompt_tokens = tokenizer.encode(prompt_text)
prompt_input = types.ModelInput.from_ints(prompt_tokens)

# 1. Генерация ответа Студентом
print("\nGenerating response...")
params = types.SamplingParams(max_tokens=1024, temperature=0.7)
result = sampling_client.sample(prompt=prompt_input, sampling_params=params, num_samples=1).result()

response_seq = result.sequences[0]
response_text = tokenizer.decode(response_seq.tokens)
print(f"\n--- Distilled Student Response ---\n{response_text}")

# 2. Оценка ответа Учителем (Logprobs)
print("\nTeacher is evaluating...")
full_tokens = prompt_tokens + list(response_seq.tokens)
full_input = types.ModelInput.from_ints(full_tokens)

teacher_logprobs_full = teacher_client.compute_logprobs(full_input).result()
response_logprobs = teacher_logprobs_full[len(prompt_tokens):]

print("\n--- Evaluation Summary ---")
print(f"Response length: {len(response_logprobs)} tokens")
print(f"Mean Teacher Logprob: {np.mean(response_logprobs):.4f}")
print(f"Confidence (exp): {np.exp(np.mean(response_logprobs))*100:.2f}%")
