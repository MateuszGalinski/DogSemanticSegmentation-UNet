from torch.utils.data import Dataset
from PIL import Image
import os

class DogDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(os.path.join(dataset_dir, 'Images')) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.dataset_dir, 'Images', self.image_filenames[idx])
        label_name = os.path.join(self.dataset_dir, 'Labels', f'annotated_{self.image_filenames[idx]}')

        image = Image.open(img_name).convert('RGB')
        label = Image.open(label_name).convert('L')  # Assuming label is a grayscale image

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label