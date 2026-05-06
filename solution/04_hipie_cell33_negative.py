# Cell 33: прогон HIPIE с НЕРЕЛЕВАНТНЫМИ промптами
#
# Идея:
#   - для каждого семпла делаем 7 "чужих" промптов (из других картинок + generic)
#   - прогоняем HIPIE на этих промптах и считаем IoU с GT
#   - ожидаем, что модель всё равно что-то сегментирует (это и есть проблема)


def run_hipie_negative_prompts(
    samples: List[ReferringItem],
    dataset: ReferringDataset,
    existing_indices: List[int],
    pred_threshold: float = 0.5,
    seed: int = 42,
) -> List[Tuple[Image.Image, np.ndarray, List[Tuple[str, np.ndarray, float]], str]]:
    """Per sample: GT mask + NUM_UNRELATED_PROMPTS HIPIE masks from unrelated text.

    Returns list of (image, gt_bool, [(prompt, pred_bool, iou), ...×7], true_caption).
    """
    rng = random.Random(seed)
    pool_items = [dataset[i] for i in existing_indices]

    out = []

    for item in tqdm(samples, desc="HIPIE negative prompts"):
        # 1. Картинка
        image = Image.open(item.image_path).convert("RGB")
        w, h = image.size

        # 2. GT маска (по true caption)
        gt_mask = _polygons_to_mask(item.segmentation, h, w)

        # 3. Истинный промпт (для подписи на графике)
        true_caption = item.captions[0] if item.captions else ""

        # 4. Семь нерелевантных промптов
        neg_prompts = _seven_unrelated_prompts(item, pool_items, rng=rng)

        # 5. Гоним HIPIE на каждом
        per_prompt = []
        for prompt in neg_prompts:
            pred_mask = get_segmentation_mask(image, prompt)
            iou_score = _iou(pred_mask, gt_mask)
            per_prompt.append((prompt, pred_mask, iou_score))

        out.append((image, gt_mask, per_prompt, true_caption))

    return out
