# Cell 69: прогоняем LISA на одном элементе датасета.
#
# ValDataset[ind] возвращает кортеж:
#   (image_path, image_clip, image_sam, conversations, masks, ...)
# Для разных версий поля могут чуть отличаться, поэтому распаковываем
# первые два элемента и берём всё нужное по индексу.


@torch.no_grad()
def infer_on_dataset(dataset, ind, params=PARAMETERS):
    # TODO: Finish implementation

    item = dataset[ind]
    # ValDataset обычно возвращает:
    # image_path, image_clip, image_sam, conversations, masks, label,
    # resize_list, questions, sampled_classes
    image_path = item[0]
    conversations = item[3]

    results = []
    for prompt in conversations:
        text_out, pred_masks = run_inference(prompt, image_path, params=params)
        results.append((text_out, pred_masks))

    return results
