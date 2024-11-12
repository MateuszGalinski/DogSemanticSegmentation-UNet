import torch
from torchmetrics.classification import JaccardIndex, BinaryAccuracy  # Import for IoU and accuracy

def evaluate_model(model, dataloader, device, threshold=0.5):
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
    iou_metric = JaccardIndex(num_classes=2, task="binary").to(device)  # Initialize JaccardIndex for IoU
    accuracy_metric = BinaryAccuracy().to(device)  # Initialize BinaryAccuracy for pixel-wise accuracy

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            # Model inference
            predicted_masks = model(images)

            # Binarize the predicted mask
            pred_binary = (predicted_masks > threshold).float()

            # Update the IoU and accuracy metrics
            iou_metric.update(pred_binary, masks.int())
            accuracy_metric.update(pred_binary, masks.int())

    # Calculate mean IoU and mean accuracy over all images in the dataset
    mean_iou = iou_metric.compute().item()  # Final IoU score
    mean_accuracy = accuracy_metric.compute().item()  # Final accuracy score

    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    
    return mean_iou, mean_accuracy
