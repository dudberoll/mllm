# Cell 53: токенизатор LISA
#
# Что делаем:
#   - грузим обычный токенизатор LLaMA из LISA-репо
#   - добавляем спец-токен [SEG]
#   - добавляем токены <im_start>, <im_end> (нужны LLaVA)
#   - запоминаем id токена [SEG] (нужен в инференсе)

VERSION = "xinlai/LISA-13B-llama2-v1"

# TODO: Load tokenizer here
tokenizer = AutoTokenizer.from_pretrained(
    VERSION,
    cache_dir=None,
    model_max_length=512,
    padding_side="right",
    use_fast=False,
)

# обычный паддинг
tokenizer.pad_token = tokenizer.unk_token

# Добавляем [SEG] и токены image-start/image-end
tokenizer.add_tokens("[SEG]")
seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

tokenizer.add_tokens(
    [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
)
