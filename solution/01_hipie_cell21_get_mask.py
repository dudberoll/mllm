# Cell 21: получить маску по картинке и текстовому промпту через HIPIE
#
# Идея:
#   1. Один раз создаём PartSegmDemo (он внутри строит модель и грузит веса).
#   2. Внутри функции прогоняем картинку + текст и берём предсказанную маску.
#   3. Возвращаем булев numpy-массив той же высоты/ширины, что и картинка.

from detectron2.projects.hipie.demo_lib.part_segm_demo import PartSegmDemo


# ---- Параметры ----
# Веса и конфиг скачали выше в ноутбуке.
# Имя конфига взято из репо HIPIE (config для r50_parts).
CONFIG_FILE = "projects/HIPIE/configs/r50_parts.yaml"
WEIGHTS_PATH = "/kaggle/working/HIPIE/pretrained_weights/r50_parts.pth"
PRED_THRESHOLD = 0.5

# Создаём демо один раз (тяжёлая операция: грузит модель и веса)
demo = PartSegmDemo(
    config_file=CONFIG_FILE,
    weights=WEIGHTS_PATH,
    score_thresh=PRED_THRESHOLD,
)


def get_segmentation_mask(image: Image.Image, prompt: str) -> np.ndarray:
    # TODO: YOUR CODE HERE

    # PIL → numpy в BGR (HIPIE/detectron2 ожидают BGR)
    image_np = convert_PIL_to_numpy(image, format="BGR")

    # Прогоняем через демо. Возвращает predictions с pred_masks.
    predictions = demo.run_on_image(image_np, prompt)

    # pred_masks: тензор формы (N, H, W) — N инстансов
    pred_masks = predictions["instances"].pred_masks.cpu().numpy()

    # Если ничего не нашлось — возвращаем пустую маску
    if len(pred_masks) == 0:
        h, w = image_np.shape[:2]
        out = np.zeros((h, w), dtype=bool)
        return out

    # Берём первый инстанс (он же самый уверенный)
    out = pred_masks[0].astype(bool)

    # raise NotImplementedError
    return out


mask = get_segmentation_mask(image, prompt)
