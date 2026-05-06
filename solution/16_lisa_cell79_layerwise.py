# Cell 79: layer-wise probing — для каждого слоя LLaMA проецируем
# [SEG]-эмбеддинг через text_hidden_fcs и декодируем маску через SAM.
#
# Шаги:
#   1. Прогоняем full forward с output_hidden_states=True.
#   2. Находим позицию [SEG] (после генерации).
#   3. На каждом из ~40 слоёв берём seg_emb, проецируем, отдаём в SAM-декодер.
#   4. Считаем IoU с GT.


def _decode_mask_with_sam(seg_emb_proj, image_sam, resize_shape, original_size):
    """Прогнать готовый prompt-эмбеддинг через SAM-декодер."""
    # 1. Получаем image-embedding через SAM-encoder
    image_embeddings = model.get_visual_embs(image_sam)

    # 2. Готовим sparse / dense prompt embeddings (только sparse от [SEG])
    sparse_emb = seg_emb_proj.unsqueeze(1)  # (1, 1, 256)
    dense_emb = model.model.visual_model.prompt_encoder.no_mask_embed.weight.reshape(
        1, -1, 1, 1
    ).expand(sparse_emb.shape[0], -1, 64, 64)

    # 3. Декодер
    low_res_masks, _ = model.model.visual_model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.model.visual_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
    )

    # 4. Resize до оригинального размера
    pred_mask = model.model.visual_model.postprocess_masks(
        low_res_masks,
        input_size=resize_shape,
        original_size=original_size,
    )
    return pred_mask[:, 0]  # (1, H, W)


@torch.no_grad()
def layerwise_probing(dataset, ind, store_all_masks=False):
    """Probe each LLM layer's [SEG] embedding and decode to mask via SAM."""

    # TODO: Finish implementation

    # 1. Берём элемент
    item = dataset[ind]
    image_path = item[0]
    conversations = item[3]
    masks = item[4]
    gt_mask = torch.from_numpy(np.array(masks[0])).cuda()

    prompt = conversations[0]

    # 2. Готовим картинку
    image_bgr = cv2.imread(image_path)
    image_rgb = image_bgr[..., ::-1]
    original_size = image_rgb.shape[:2]
    image_clip = preprocess_clip(image_rgb)
    image_sam, resize_shape = preprocess_sam(image_rgb)

    # 3. Делаем .evaluate, чтобы получить output_ids (с [SEG])
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

    # 4. Полный forward с hidden states по всем слоям
    full_ids = torch.cat([input_ids[0], output_ids[0]], dim=0).unsqueeze(0)
    outputs = model(
        images=image_sam,
        images_clip=image_clip,
        input_ids=full_ids,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states  # tuple длины num_layers + 1
    num_layers = len(hidden_states)

    # 5. Находим позицию [SEG]
    seg_positions = (full_ids[0] == seg_token_idx).nonzero(as_tuple=True)[0]
    if len(seg_positions) == 0:
        return None
    seg_pos = seg_positions[-1].item()

    # 6. Перебираем слои
    proj = model.text_hidden_fcs[0]
    ious = []
    layer_masks = []

    for layer_idx in range(num_layers):
        seg_emb = hidden_states[layer_idx][0, seg_pos]  # (D,)
        seg_emb_proj = proj(seg_emb.unsqueeze(0).to(torch_dtype))  # (1, 256)

        pred_mask = _decode_mask_with_sam(
            seg_emb_proj, image_sam, resize_shape, original_size
        )  # (1, H, W)

        iou_score = calculate_iou(pred_mask[0], gt_mask)
        ious.append(iou_score)

        if store_all_masks:
            layer_masks.append((pred_mask[0] > 0).float().cpu().numpy())
        else:
            layer_masks.append(None)

    return {
        "ious": ious,
        "layer_masks": layer_masks,
        "gt_mask": gt_mask.cpu().numpy(),
        "image_path": image_path,
        "num_layers": num_layers,
        "conversations": conversations,
    }
