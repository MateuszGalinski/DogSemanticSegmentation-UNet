from torch.utils.data import Dataset
from PIL import Image
import os
import re

class DogDataset(Dataset):
    def __init__(self, dataset_dir, indexes_below, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform

        self.image_filenames = [
            f for f in os.listdir(os.path.join(dataset_dir, 'Images'))
            if f.endswith('.jpg') and self._is_index_less_than(f, indexes_below)
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.dataset_dir, 'Images', self.image_filenames[idx])
        label_filename = self.image_filenames[idx].replace('.jpg', '.png')
        label_name = os.path.join(self.dataset_dir, 'Labels', label_filename)

        image = Image.open(img_name).convert('RGB')
        label = Image.open(label_name).convert('L')  # Assuming label is a grayscale image

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

    def _is_index_less_than(self, filename, threshold):
        # Use regex to find the number at the end of the filename (before the file extension)
        match = re.search(r'_(\d+)\.jpg$', filename)
        if match:
            number = int(match.group(1))
            return number < threshold
        return False
