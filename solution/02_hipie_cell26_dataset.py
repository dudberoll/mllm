# Cell 26: класс датасета RefCOCO
#
# Что хранится:
#   refs(google).p — pickle, список словарей:
#       {'ref_id', 'ann_id', 'image_id', 'split', 'sentences': [{'raw': '...'}, ...]}
#   instances.json — стандартный COCO json:
#       {'annotations': [{'id', 'image_id', 'bbox', 'segmentation'}, ...]}
#
# Главная связка: ref['ann_id'] -> annotation['id']

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import pickle


@dataclass
class ReferringItem:
    """Single item from RefCOCO dataset."""
    image_path: Path
    captions: List[str]
    bbox: List[float]
    segmentation: List[List[float]]
    image_id: int = None
    ann_id: int = None


class ReferringDataset:
    """RefCOCO dataset wrapper for referring expression segmentation."""

    def __init__(self, data_path: Path, split: str = None, label_type: str = 'google'):
        # TODO: Finish implementation
        self.data_path = Path(data_path)
        self.train_img_path = self.data_path / "train2014"
        self.split = split
        self.label_type = label_type
        self.refs, self.bbox_lookup, self.segmentation_lookup = self._load_data()
        print(f"Loaded {len(self.refs)} referring expressions (split={split})")

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, idx: int) -> ReferringItem:
        # TODO: Finish implementation
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")

        ref = self.refs[idx]
        image_id = ref["image_id"]
        ann_id = ref["ann_id"]

        # Все captions у одного ref-а
        captions = []
        for sent in ref.get("sentences", []):
            captions.append(sent["raw"])

        bbox = self.bbox_lookup.get(ann_id, [])
        segmentation = self.segmentation_lookup.get(ann_id, [])
        image_path = self.get_image_path(image_id)

        return ReferringItem(
            image_path=image_path,
            captions=captions,
            bbox=bbox,
            segmentation=segmentation,
            image_id=image_id,
            ann_id=ann_id,
        )

    def _load_annotations(self, path: Path):
        """Прочитать pickle/json в зависимости от расширения."""
        path = Path(path)
        if path.suffix == ".p":
            with open(path, "rb") as f:
                return pickle.load(f)
        if path.suffix == ".json":
            with open(path, "r") as f:
                return json.load(f)
        raise ValueError(f"Unknown extension: {path.suffix}")

    def _load_data(self) -> Tuple[List[Dict], Dict, Dict]:
        # TODO: Finish implementation
        refs = self._load_annotations(self.data_path / f'refcoco/refs({self.label_type}).p')
        if self.split is not None:
            refs = [ref for ref in refs if ref['split'] == self.split]

        instances = self._load_annotations(self.data_path / "refcoco/instances.json")

        # Строим словари ann_id -> bbox / segmentation
        bbox_lookup = {}
        segmentation_lookup = {}
        for ann in instances["annotations"]:
            ann_id = ann["id"]
            bbox_lookup[ann_id] = ann.get("bbox", [])
            segmentation_lookup[ann_id] = ann.get("segmentation", [])

        return refs, bbox_lookup, segmentation_lookup

    def get_image_path(self, image_id: int) -> Path:
        return self.train_img_path / f'COCO_train2014_{image_id:012d}.jpg'

    def image_file_exists(self, idx: int) -> bool:
        """Есть ли картинка на диске для этого индекса."""
        item = self[idx]
        return item.image_path.exists()

    def list_indices_with_images_on_disk(self) -> List[int]:
        """Все индексы, для которых картинка реально лежит на диске."""
        good_indices = []
        for idx in range(len(self)):
            ref = self.refs[idx]
            image_id = ref["image_id"]
            image_path = self.get_image_path(image_id)
            if image_path.exists():
                good_indices.append(idx)
        return good_indices
