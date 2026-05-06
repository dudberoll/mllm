# Cell 80: бейзлайн — обычный прогон LISA + IoU.

def get_baseline_prediction(dataset, ind, gt_mask):
    """Run standard LISA inference and compute IoU with ground truth."""
    # TODO: Finish implementation

    item = dataset[ind]
    image_path = item[0]
    conversations = item[3]

    prompt = conversations[0]
    text_out, pred_masks = run_inference(prompt, image_path)

    if pred_masks is None or len(pred_masks) == 0:
        h, w = gt_mask.shape if isinstance(gt_mask, np.ndarray) else gt_mask.shape
        baseline_mask = np.zeros((h, w), dtype=bool)
        return 0.0, baseline_mask

    # pred_masks[0] — torch tensor (1, H, W)
    pm = pred_masks[0]
    if isinstance(pm, torch.Tensor):
        pm = pm.detach().cpu().numpy()
    if pm.ndim == 3:
        pm = pm[0]

    baseline_mask = pm > 0

    # gt_mask может быть numpy или tensor — приводим к numpy
    if isinstance(gt_mask, torch.Tensor):
        gt_np = gt_mask.cpu().numpy().astype(bool)
    else:
        gt_np = np.asarray(gt_mask).astype(bool)

    inter = np.logical_and(baseline_mask, gt_np).sum()
    union = np.logical_or(baseline_mask, gt_np).sum()
    baseline_iou = float(inter / union) if union > 0 else 0.0

    return baseline_iou, baseline_mask
