# Cell 54: загрузка LISA в 4-битном режиме
#
# Зачем 4-bit:
#   LISA-13B в fp16 ~26 ГБ — не влезет в T4 (16 ГБ).
#   В 4-bit веса ~7 ГБ — нормально.

# TODO: Load model in 4-bit mode here
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

torch_dtype = torch.float16

model = LISAForCausalLM.from_pretrained(
    VERSION,
    torch_dtype=torch_dtype,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    seg_token_idx=seg_token_idx,
)

# config для LLaVA
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
