# Cell 57: параметры инференса LISA

# TODO: Define used parameters of the model here
PARAMETERS = {
    "image_size_clip": 224,         # CLIP вход
    "image_size_sam": 1024,         # SAM вход
    "max_new_tokens": 512,          # сколько токенов разрешаем сгенерить
    "conv_type": "llava_v1",        # тип шаблона диалога LLaVA
    "use_mm_start_end": True,       # вставлять <im_start>...<im_end>
    "pixel_mean": [0.48145466, 0.4578275, 0.40821073],
    "pixel_std":  [0.26862954, 0.26130258, 0.27577711],
    # SAM нормализация
    "sam_pixel_mean": [123.675, 116.28, 103.53],
    "sam_pixel_std":  [58.395, 57.12, 57.375],
}
