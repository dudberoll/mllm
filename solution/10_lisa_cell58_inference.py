# Cell 58: главный пайплайн инференса LISA
#
# Шаги внутри run_inference:
#   1. Читаем картинку (cv2 BGR -> RGB).
#   2. Готовим CLIP-вход (224x224, нормализация CLIP).
#   3. Готовим SAM-вход (resize до 1024, padding до 1024x1024, нормализация).
#   4. Строим conversation в формате LLaVA: USER: <image>\n{prompt} ASSISTANT:
#   5. Токенизируем (с заменой <image> на IMAGE_TOKEN_INDEX).
#   6. Зовём model.evaluate(...) — это уже определено в LISA.
#   7. Декодируем токены в текст. Возвращаем (text, pred_masks).


def preprocess_clip(image_rgb, params=PARAMETERS):
    """Готовим картинку для CLIP."""
    image_clip = clip_image_processor.preprocess(image_rgb, return_tensors="pt")["pixel_values"][0]
    image_clip = image_clip.unsqueeze(0).cuda().to(torch_dtype)
    return image_clip


def preprocess_sam(image_rgb, params=PARAMETERS):
    """Готовим картинку для SAM: resize -> normalize -> pad до 1024x1024."""
    image_size_sam = params["image_size_sam"]

    # 1. Resize длинной стороны до 1024
    image_sam = transform.apply_image(image_rgb)
    resize_shape = image_sam.shape[:2]  # (h_new, w_new) до паддинга

    # 2. В тензор (C, H, W), float
    image_sam = torch.from_numpy(image_sam).permute(2, 0, 1).contiguous().float()

    # 3. Нормализация (как в SAM)
    pixel_mean = torch.tensor(params["sam_pixel_mean"]).view(-1, 1, 1)
    pixel_std = torch.tensor(params["sam_pixel_std"]).view(-1, 1, 1)
    image_sam = (image_sam - pixel_mean) / pixel_std

    # 4. Padding до 1024x1024 (справа и снизу нулями)
    h, w = image_sam.shape[-2:]
    pad_h = image_size_sam - h
    pad_w = image_size_sam - w
    image_sam = F.pad(image_sam, (0, pad_w, 0, pad_h))

    image_sam = image_sam.unsqueeze(0).cuda().to(torch_dtype)
    return image_sam, resize_shape


def build_prompt(prompt: str, params=PARAMETERS) -> str:
    """Сборка диалога в формате LLaVA."""
    conv = conversation_lib.conv_templates[params["conv_type"]].copy()
    conv.messages = []

    # Подставляем токены image-start/end
    if params["use_mm_start_end"]:
        image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    else:
        image_token = DEFAULT_IMAGE_TOKEN

    user_message = image_token + "\n" + prompt
    conv.append_message(conv.roles[0], user_message)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


@torch.no_grad()
def run_inference(prompt, image_path, params=PARAMETERS):
    # TODO: Finish implementation

    # 1. Картинка
    image_bgr = cv2.imread(image_path)
    image_rgb = image_bgr[..., ::-1]  # BGR -> RGB
    original_size = image_rgb.shape[:2]  # (H, W)

    # 2. Подготовка для CLIP и SAM
    image_clip = preprocess_clip(image_rgb, params)
    image_sam, resize_shape = preprocess_sam(image_rgb, params)

    # 3. Промпт
    full_prompt = build_prompt(prompt, params)
    input_ids = tokenizer_image_token(full_prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    # 4. Запуск (LISA знает, что делать через .evaluate)
    output_ids, pred_masks = model.evaluate(
        image_clip,
        image_sam,
        input_ids,
        [resize_shape],
        [original_size],
        max_new_tokens=params["max_new_tokens"],
        tokenizer=tokenizer,
    )

    # 5. Декодирование текста
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)

    return text_output, pred_masks
