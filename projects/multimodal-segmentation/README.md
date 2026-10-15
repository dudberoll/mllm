# Text-Guided Multimodal Segmentation

Experiments with two ways to segment images from language prompts:

- **HIPIE + RefCOCO:** referring-expression segmentation, mask IoU evaluation, and negative prompts.
- **LISA + ReasonSeg:** text-only and segmentation responses, `[SEG]` token behavior, and layer-wise probing of its representation.

The complete notebook is [`notebooks/text_guided_segmentation.ipynb`](notebooks/text_guided_segmentation.ipynb). The HIPIE implementation is in [`../HIPIE/`](../HIPIE/); selected extracted solution cells are in [`../../solution/`](../../solution/).

Re-running the notebook requires the pretrained model checkpoints, datasets, and a CUDA-capable environment. Those model weights are not included here.
