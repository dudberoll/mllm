# Cell 77: IoU для тензоров.

from utils.utils import intersectionAndUnionGPU


def calculate_iou(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    # TODO: Finish implementation
    pred_bool = (y_pred > 0).bool()
    true_bool = y_true.bool()

    inter = torch.logical_and(pred_bool, true_bool).sum().item()
    union = torch.logical_or(pred_bool, true_bool).sum().item()

    if union == 0:
        return 0.0
    return inter / union
