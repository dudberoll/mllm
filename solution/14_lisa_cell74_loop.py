# Cell 74: визуализация [SEG] similarity на нескольких семплах.

sample_indices = [5, 9, 11, 16, 19]

for ind in sample_indices:
    # TODO: Finish implementation
    item = train_dataset[ind]
    image_path = item[0]
    conversations = item[3]
    masks = item[4]

    image_bgr = cv2.imread(image_path)
    image_np = image_bgr[..., ::-1]

    # Берём первый промпт из conversations
    prompt = conversations[0]

    similarity_overlay, pred_masks = compute_seg_similarity(
        prompt=prompt,
        image_path=image_path,
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(square_pad(image_np))
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(square_pad(masks[0]), cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    if pred_masks and pred_masks[0].numel() > 0:
        axes[2].imshow(overlay_mask(image_np, pred_masks[0]))
        axes[2].set_title("LISA Prediction")
        axes[2].axis("off")
    else:
        axes[2].set_visible(False)

    axes[3].imshow(similarity_overlay)
    axes[3].set_title("[SEG] Similarity")
    axes[3].axis("off")

    fig.suptitle(f"Train sample {ind} — {conversations[0][:80]}...", fontsize=11)
    plt.tight_layout()
    plt.show()
