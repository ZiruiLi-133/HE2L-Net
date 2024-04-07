import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms

class MathExpressionDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        """
        Args:
            json_file (string): Path to the JSON file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open(json_file) as f:
            self.annotations = json.load(f)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        # Get the image info
        img_id = self.annotations['images'][idx]['id']
        img_name = os.path.join(self.img_dir, self.annotations['images'][idx]['file_name'])
        image = Image.open(img_name).convert('RGB')

        # Get annotations
        annotations = [a for a in self.annotations['annotations'] if a['image_id'] == img_id]
        boxes = torch.as_tensor([a['bbox'] for a in annotations], dtype=torch.float32)
        labels = torch.as_tensor([a['category_id'] for a in annotations], dtype=torch.int64)
        labels_str = [self.annotations['categories'][label.item() - 1]['name'] for label in labels]
        print(f'label_str in getitem: {labels_str}')
        # Your transform here, for example, converting PIL image to tensor.
        if self.transform:
            image = self.transform(image)

        # Return the image, the bounding boxes, and the labels
        sample = {'image': image, 'boxes': boxes, 'labels': labels, 'labels_str': labels_str}

        return sample


def custom_collate_fn(batch):
    """
    Custom collate function to handle batches, specifically ensuring labels_str
    is in a list of lists of strings format.

    Args:
        batch: List of tuples from Dataset.__getitem__()

    Returns:
        A batch with images, boxes, labels, and labels_str properly collated.
    """
    # Separate the components of the batch
    images = [item['image'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    labels_str = [item['labels_str'] for item in batch]  # No need to modify this

    # Use default_collate for images, boxes, and labels, which are typically tensors
    collated_images = default_collate(images)
    collated_boxes = default_collate(boxes)
    collated_labels = default_collate(labels)

    # No modification needed for labels_str, it's already in the desired format
    # However, you can enforce or check the format here if necessary

    # Construct and return the final batch as a dictionary
    return {
        'image': collated_images,
        'boxes': collated_boxes,
        'labels': collated_labels,
        'labels_str': labels_str
    }

def get_calc_dataloader(batch_size, shuffle=True, num_workers=1):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor.
    ])

    # Create the dataset
    dataset = MathExpressionDataset(json_file='./kaggle_data_coco.json',
                                    img_dir='D:\\Machine_Learning_Projects\\Datasets\\handwriting_calculas\\batch_1\\background_images',
                                    transform=transform)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=custom_collate_fn)
    return dataloader