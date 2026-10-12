## SFT rank-128 merge bundle

В этой папке лежит минимальный набор для слияния скачанного Tinker adapter с базовой моделью Hugging Face.

Содержимое:
- `merge_tinker_adapter_to_hf_model.py` — скрипт слияния
- `adapter_weights/adapter_model.safetensors` — LoRA adapter
- `adapter_weights/adapter_config.json` — конфиг adapter

Базовая модель для слияния:
- `Qwen/Qwen3-8B-Base`

Пример запуска:

```bash
python merge_tinker_adapter_to_hf_model.py \
  --hf-model Qwen/Qwen3-8B-Base \
  --tinker-adapter-path ./adapter_weights \
  --output-path ./merged_qwen3_8b_sft_rank128
```

После этого папку `merged_qwen3_8b_sft_rank128` можно использовать как обычную Hugging Face модель, в том числе в Colab.
