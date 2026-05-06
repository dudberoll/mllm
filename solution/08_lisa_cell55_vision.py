# Cell 55: загрузка vision-модулей (CLIP image processor + SAM vision tower)

# TODO: Load vision modules
# Инициализируем vision (CLIP) — внутри грузится CLIP encoder
model.get_model().initialize_vision_modules(model.get_model().config)
vision_tower = model.get_model().get_vision_tower()
vision_tower.to(dtype=torch_dtype)

# CLIP image processor (нужен, чтобы готовить картинку 224х224)
clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)

# SAM transform (resize длинной стороны до 1024)
transform = ResizeLongestSide(1024)

# Переводим всё на GPU
model = model.cuda()
vision_tower = vision_tower.cuda()

model.eval()
