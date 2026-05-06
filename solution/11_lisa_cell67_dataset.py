# Cell 67: датасет ReasonSeg.
#
# В репо LISA есть готовый ValDataset, который умеет читать ReasonSeg.

# TODO: Define ReasonSeg Dataset here
from utils.dataset import ValDataset

BASE_DIR = "/kaggle/working/LISA/data"
IMAGE_SIZE = 1024

val_dataset = ValDataset(
    base_image_dir=BASE_DIR,
    tokenizer=tokenizer,
    vision_tower=model.config.vision_tower,
    val_dataset="ReasonSeg|val",
    image_size=IMAGE_SIZE,
)

train_dataset = ValDataset(
    base_image_dir=BASE_DIR,
    tokenizer=tokenizer,
    vision_tower=model.config.vision_tower,
    val_dataset="ReasonSeg|train",
    image_size=IMAGE_SIZE,
)

test_dataset = ValDataset(
    base_image_dir=BASE_DIR,
    tokenizer=tokenizer,
    vision_tower=model.config.vision_tower,
    val_dataset="ReasonSeg|test",
    image_size=IMAGE_SIZE,
)

print(f"Training samples: {len(train_dataset)}, "
      f"Validation samples: {len(val_dataset)}, "
      f"Test samples: {len(test_dataset)}")
