import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_segmentation(model, dataloader, device):
    """
    Visualize a random image from the test dataset along with the ground truth and predicted segmentation mask.

    Args:
    - model (nn.Module): The trained model.
    - dataloader (DataLoader): DataLoader for the test dataset.
    - device (torch.device): The device (CPU or CUDA) to run the evaluation on.
    """
    model.eval()  # Set the model to evaluation mode

    # Get a sample from the test set
    data_images, label_images = next(iter(dataloader))  # Get the first batch
    data_images, label_images = data_images.to(device), label_images.to(device)

    # Pick a random index from the batch to visualize
    idx = np.random.randint(0, data_images.size(0))  # Random index from batch
    input_image = data_images[idx].unsqueeze(0)  # Select a single image (batch dimension added)
    true_mask = label_images[idx].unsqueeze(0)   # Corresponding true mask

    # Forward pass: Get model's prediction for the input image
    with torch.no_grad():  # No gradients needed for inference
        output = model(input_image)  # Model prediction
        output = torch.argmax(output, dim=1).cpu()  # Get predicted class by selecting max logit (segmentation mask)

    # Convert tensors to numpy for visualization
    input_image = input_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # CHW to HWC format
    true_mask = true_mask.squeeze(0).cpu().numpy()  # True segmentation mask (label)
    predicted_mask = output.squeeze(0).cpu().numpy()  # Predicted segmentation mask

    # Plotting the images
    figure, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Plot original image
    ax[0].imshow(input_image)
    ax[0].set_title("Input Image")
    ax[0].axis('off')

    # Plot true mask (ground truth)
    ax[1].imshow(true_mask, cmap="gray")
    ax[1].set_title("True Mask")
    ax[1].axis('off')

    # Plot predicted mask
    ax[2].imshow(predicted_mask, cmap="gray")
    ax[2].set_title("Predicted Mask")
    ax[2].axis('off')

    plt.show()
