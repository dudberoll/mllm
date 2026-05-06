# Cell 73: визуализация attention [SEG]-токена.
#
# Идея:
#   - LLM выдаёт hidden states для всех токенов.
#   - 256 image-токенов = CLIP-патчи (16x16 = 256), пропущенные через LLM.
#   - [SEG]-токен — последний нужный токен.
#   - Берём оба, проецируем через text_hidden_fcs (тот самый MLP,
#     который превращает [SEG] в prompt-эмбеддинг для SAM).
#   - Считаем cos(seg_emb, patch_emb) для каждого патча -> heatmap 16x16.
#   - Апсемплим до картинки и накладываем как overlay.


def square_pad(image_np):
    """Дополнить картинку нулями до квадрата (для красивого вывода)."""
    h, w = image_np.shape[:2]
    side = max(h, w)
    if image_np.ndim == 2:
        out = np.zeros((side, side), dtype=image_np.dtype)
        out[:h, :w] = image_np
    else:
        out = np.zeros((side, side, image_np.shape[2]), dtype=image_np.dtype)
        out[:h, :w] = image_np
    return out


@torch.no_grad()
def compute_seg_similarity(prompt, image_path, params=PARAMETERS):
    # TODO: Finish implementation

    # 1. Готовим картинку
    image_bgr = cv2.imread(image_path)
    image_rgb = image_bgr[..., ::-1]
    original_size = image_rgb.shape[:2]

    image_clip = preprocess_clip(image_rgb, params)
    image_sam, resize_shape = preprocess_sam(image_rgb, params)

    # 2. Получаем маски и токены через model.evaluate
    full_prompt = build_prompt(prompt, params)
    input_ids = tokenizer_image_token(full_prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    output_ids, pred_masks = model.evaluate(
        image_clip,
        image_sam,
        input_ids,
        [resize_shape],
        [original_size],
        max_new_tokens=params["max_new_tokens"],
        tokenizer=tokenizer,
    )

    # 3. Полный forward, чтобы вытащить hidden states
    # Склеиваем input_ids + output_ids (то, что сгенерировано)
    full_ids = torch.cat([input_ids[0], output_ids[0]], dim=0).unsqueeze(0)

    outputs = model(
        images=image_sam,
        images_clip=image_clip,
        input_ids=full_ids,
        output_hidden_states=True,
        return_dict=True,
    )

    last_hidden = outputs.hidden_states[-1]  # (1, T, D)

    # 4. Находим позицию [SEG]
    seg_positions = (full_ids[0] == seg_token_idx).nonzero(as_tuple=True)[0]
    if len(seg_positions) == 0:
        # нет [SEG] — нечего показывать
        h, w = image_rgb.shape[:2]
        return np.zeros((h, w, 3), dtype=np.uint8), pred_masks
    seg_pos = seg_positions[0].item()
    seg_emb = last_hidden[0, seg_pos]  # (D,)

    # 5. Достаём 256 image-токенов.
    # LISA вставляет картинку в позицию IMAGE_TOKEN_INDEX, после паддинга
    # это превращается в 256 visual токенов. Берём их по позиции <image>.
    image_token_positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    img_start = image_token_positions[0].item()
    image_emb = last_hidden[0, img_start:img_start + 256]  # (256, D)

    # 6. Проецируем через text_hidden_fcs (один и тот же MLP)
    proj = model.text_hidden_fcs[0]
    seg_proj = proj(seg_emb.unsqueeze(0))  # (1, 256)
    image_proj = proj(image_emb)           # (256, 256)

    # 7. Cosine similarity
    sim = F.cosine_similarity(seg_proj, image_proj, dim=-1)  # (256,)
    sim = sim.float().cpu().numpy().reshape(16, 16)

    # 8. Нормируем в [0, 1] и апсемплим до размера картинки
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
    h, w = image_rgb.shape[:2]
    heatmap = cv2.resize(sim, (w, h), interpolation=cv2.INTER_CUBIC)

    # 9. Накладываем на картинку
    heatmap_color = (plt.cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.5 * image_rgb + 0.5 * heatmap_color).astype(np.uint8)

    return overlay, pred_masks
