import torch

def calculate_iou(pred_mask, true_mask, threshold=0.5):
    """
    Computes Intersection over Union (IoU) between the predicted and true masks.
    
    Args:
        pred_mask (torch.Tensor): Predicted mask from the model, expected shape [B, 1, H, W].
        true_mask (torch.Tensor): Ground truth mask, expected shape [B, 1, H, W].
        threshold (float): Threshold to binarize predictions if using BCEWithLogitsLoss.
        
    Returns:
        float: IoU score.
    """
    # Binarize the predicted mask
    pred_binary = (torch.sigmoid(pred_mask) > threshold).float()  # Binary mask [B, 1, H, W]

    # Intersection and Union calculation
    intersection = (pred_binary * true_mask).sum(dim=(1, 2, 3))  # Sum over height and width dimensions
    union = (pred_binary + true_mask - pred_binary * true_mask).sum(dim=(1, 2, 3))  # Union
    
    iou = (intersection / union).mean().item()  # Average IoU over the batch
    return iou

def calculate_accuracy(pred_mask, true_mask, threshold=0.5):
    """
    Computes pixel-wise accuracy between the predicted and true masks.
    
    Args:
        pred_mask (torch.Tensor): Predicted mask from the model, expected shape [B, 1, H, W].
        true_mask (torch.Tensor): Ground truth mask, expected shape [B, 1, H, W].
        threshold (float): Threshold to binarize predictions if using BCEWithLogitsLoss.
        
    Returns:
        float: Pixel-wise accuracy score.
    """
    # Binarize the predicted mask
    pred_binary = (torch.sigmoid(pred_mask) > threshold).float()  # Binary mask [B, 1, H, W]

    # Accuracy calculation
    correct_pixels = (pred_binary == true_mask).sum(dim=(1, 2, 3))
    total_pixels = true_mask.shape[2] * true_mask.shape[3]  # H * W
    accuracy = (correct_pixels / total_pixels).mean().item()  # Average accuracy over the batch
    return accuracy

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on the provided dataloader using IoU and accuracy metrics.
    
    Args:
        model (torch.nn.Module): The trained segmentation model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test set.
        device (torch.device): Device to perform computations on ("cpu" or "cuda").
        
    Returns:
        mean_iou : float,
        mean_accuracy : float 
    """
    model.eval()
    iou_scores = []
    accuracy_scores = []

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            # Model inference
            predicted_masks = model(images)

            # Calculate IoU and accuracy for the batch
            batch_iou = calculate_iou(predicted_masks, masks)
            batch_accuracy = calculate_accuracy(predicted_masks, masks)

            iou_scores.append(batch_iou)
            accuracy_scores.append(batch_accuracy)

    # Calculate mean metrics over all batches
    mean_iou = sum(iou_scores) / len(iou_scores)
    mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)

    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    
    return mean_iou, mean_accuracy

# # Example usage:
# # Assuming `test_dataloader` is your DataLoader for the test set and `device` is the computation device
# evaluate_model(model, test_dataloader, device)
