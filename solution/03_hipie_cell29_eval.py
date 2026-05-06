# Cell 29: посчитать качество HIPIE на RefCOCO
#
# Идея:
#   - на каждом семпле берём ПЕРВУЮ caption (item.captions[0])
#   - прогоняем HIPIE -> предсказанная маска (bool)
#   - GT маску получаем из polygon-ов через pycocotools
#   - IoU = пересечение / объединение

from typing import List, Tuple
from tqdm.auto import tqdm
from pycocotools import mask as mask_utils


def _iou(pred_bool: np.ndarray, gt_bool: np.ndarray) -> float:
    # TODO: Finish implementation
    pred_bool = pred_bool.astype(bool)
    gt_bool = gt_bool.astype(bool)

    inter = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    if union == 0:
        return 0.0
    iou = inter / union
    return float(iou)


def _polygons_to_mask(segmentation, height: int, width: int) -> np.ndarray:
    """COCO polygons -> бинарная маска (H, W)."""
    if segmentation is None or len(segmentation) == 0:
        return np.zeros((height, width), dtype=bool)

    rles = mask_utils.frPyObjects(segmentation, height, width)
    decoded = mask_utils.decode(rles)
    # decode может вернуть (H, W) или (H, W, N)
    if decoded.ndim == 3:
        gt = decoded.any(axis=-1)
    else:
        gt = decoded.astype(bool)
    return gt


def run_hipie_on_refcoco(
    samples: List[ReferringItem],
    caption_index: int = 0,
    pred_threshold: float = 0.5,
) -> List[Tuple[Image.Image, str, np.ndarray, np.ndarray, float]]:
    """For each sample: first caption → HIPIE mask; IoU vs GT polygon."""
    # TODO: Finish implementation
    results = []

    for item in tqdm(samples, desc="HIPIE on RefCOCO"):
        # 1. Загружаем картинку
        image = Image.open(item.image_path).convert("RGB")
        w, h = image.size

        # 2. Берём промпт (одна caption на семпл)
        if len(item.captions) == 0:
            continue
        prompt = item.captions[caption_index]

        # 3. Прогоняем HIPIE
        pred_mask = get_segmentation_mask(image, prompt)

        # 4. Строим GT маску из polygon
        gt_mask = _polygons_to_mask(item.segmentation, h, w)

        # 5. Считаем IoU
        iou_score = _iou(pred_mask, gt_mask)

        results.append((image, prompt, pred_mask, gt_mask, iou_score))

    return results
