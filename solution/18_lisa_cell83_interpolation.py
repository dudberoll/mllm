# Cell 83: интерполяция между [SEG]-эмбеддингами двух промптов.
#
# Идея:
#   - на одной картинке два разных промпта дают два разных [SEG]-эмбеддинга A и B.
#   - линейно мешаем: emb = (1 - alpha) * A + alpha * B
#   - декодируем каждый микс через SAM-декодер, смотрим как маска плавно
#     перетекает от объекта A к объекту B.


def _get_seg_embedding(prompt, image_path):
    """Получить итоговый [SEG]-эмбеддинг (после text_hidden_fcs) для одного промпта."""
    image_bgr = cv2.imread(image_path)
    image_rgb = image_bgr[..., ::-1]
    original_size = image_rgb.shape[:2]
    image_clip = preprocess_clip(image_rgb)
    image_sam, resize_shape = preprocess_sam(image_rgb)

    full_prompt = build_prompt(prompt)
    input_ids = tokenizer_image_token(full_prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    output_ids, _ = model.evaluate(
        image_clip,
        image_sam,
        input_ids,
        [resize_shape],
        [original_size],
        max_new_tokens=PARAMETERS["max_new_tokens"],
        tokenizer=tokenizer,
    )

    full_ids = torch.cat([input_ids[0], output_ids[0]], dim=0).unsqueeze(0)
    outputs = model(
        images=image_sam,
        images_clip=image_clip,
        input_ids=full_ids,
        output_hidden_states=True,
        return_dict=True,
    )
    last_hidden = outputs.hidden_states[-1]

    seg_positions = (full_ids[0] == seg_token_idx).nonzero(as_tuple=True)[0]
    if len(seg_positions) == 0:
        return None, image_sam, resize_shape, original_size

    seg_pos = seg_positions[-1].item()
    seg_emb = last_hidden[0, seg_pos]
    seg_emb_proj = model.text_hidden_fcs[0](seg_emb.unsqueeze(0).to(torch_dtype))

    return seg_emb_proj, image_sam, resize_shape, original_size


@torch.no_grad()
def interpolation_experiment(dataset, ind, prompt_a, prompt_b, n_steps=7):
    # TODO: Finish implementation

    item = dataset[ind]
    image_path = item[0]
    masks = item[4]

    # 1. Получаем эмбеддинги для двух промптов
    emb_a, image_sam, resize_shape, original_size = _get_seg_embedding(prompt_a, image_path)
    emb_b, _, _, _ = _get_seg_embedding(prompt_b, image_path)

    if emb_a is None or emb_b is None:
        return None

    alphas = np.linspace(0, 1, n_steps)
    interp_masks = []

    for alpha in alphas:
        emb_mix = (1 - alpha) * emb_a + alpha * emb_b
        pred_mask = _decode_mask_with_sam(
            emb_mix, image_sam, resize_shape, original_size
        )
        mask_np = (pred_mask[0] > 0).float().cpu().numpy()
        interp_masks.append(mask_np)

    return dict(
        alphas=alphas,
        interp_masks=interp_masks,
        image_path=image_path,
        gt_mask=masks[0],
        prompt_a=prompt_a,
        prompt_b=prompt_b,
    )
