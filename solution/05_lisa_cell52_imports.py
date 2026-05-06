# Cell 52: импорты для LISA
#
# Часть импортов уже сделана в шаблоне.
# Тут добавляем те, что нужны для загрузки модели и инференса.

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from pprint import pprint
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

# TODO: Add all necessary imports here
from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from model.segment_anything.utils.transforms import ResizeLongestSide
