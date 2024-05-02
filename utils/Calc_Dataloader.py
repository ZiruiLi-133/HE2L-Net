import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from configs import cfgs

class MathExpressionDataset(Dataset):
    def __init__(self, split, transform=None):
        """
        Args:
            json_file (string): Path to the JSON file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if split == 'train':
            json_file = 'calculus_train.json'
        elif split == 'val':
            json_file = 'calculus_val.json'
        elif split == 'test':
            json_file = 'calculus_test.json'
        json_file_path = os.path.join(cfgs.TRAIN_Com.DATASET.root, json_file)
        with open(json_file_path) as f:
            self.annotations = json.load(f)
        self.img_dir = cfgs.TRAIN_Com.DATASET.image_root
        self.transform = transforms.Compose([
                                            transforms.ToTensor(),  # Convert image to tensor.
                                            ])

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        # Get the image info
        img_id = self.annotations['images'][idx]['id']
        latex_code = self.annotations['images'][idx]['latex']
        img_name = os.path.join(self.img_dir, self.annotations['images'][idx]['file_name'])
        img_width = self.annotations['images'][idx]['width']
        img_height = self.annotations['images'][idx]['height']
        img_size = (img_width, img_height)
        image = Image.open(img_name).convert('RGB')

        # Get annotations
        annotations = [a for a in self.annotations['annotations'] if a['image_id'] == img_id]

        boxes = torch.as_tensor([a['bbox'] for a in annotations], dtype=torch.float32)
        if(cfgs.TRAIN_Com.normalize_bbox):
            boxes[:, 0] = boxes[:, 0] / img_width  # Normalize x_min
            boxes[:, 1] = boxes[:, 1] / img_height  # Normalize y_min
            boxes[:, 2] = boxes[:, 2] / img_width  # Normalize width
            boxes[:, 3] = boxes[:, 3] / img_height  # Normalize height
        labels = torch.as_tensor([a['category_id'] for a in annotations], dtype=torch.int64)
        labels_str = [self.annotations['categories'][label.item() - 1]['name'] for label in labels]
        # print(f'label_str in getitem: {labels_str}')

        image = self.transform(image)
        # print(f'image shape in getitem: {image.shape}')
        # Return the image, the bounding boxes, and the labels
        sample = {'image': image, 'image_size': img_size,
                  'latex_code': latex_code, 'boxes': boxes,
                  'labels': labels, 'labels_str': labels_str}
        return sample


def latex_parser(latex_code):
    # latex_code: string of latex code
    symbol_to_id_path = './symbol_to_id.json'
    with open(symbol_to_id_path) as f:
        symbol_to_id = json.load(f)
    print(symbol_to_id)

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
    image = [item['image'] for item in batch]
    image_size = batch[0]['image_size']
    latex_code = [item['latex_code'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    labels_str = [item['labels_str'] for item in batch]  # No need to modify this

    # Use default_collate for images, boxes, and labels, which are typically tensors
    collated_images = default_collate(image)
    collated_boxes = default_collate(boxes)
    collated_labels = default_collate(labels)
    collated_latex = default_collate(latex_code)

    # Construct and return the final batch as a dictionary
    return {
        'image': collated_images,
        'image_size': image_size,
        'latex_code': collated_latex,
        'boxes': collated_boxes,
        'labels': collated_labels,
        'labels_str': labels_str
    }

def get_calc_dataloader(batch_size, split='train', shuffle=True, num_workers=1):
    # Create the dataset
    dataset = MathExpressionDataset(split=split)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=custom_collate_fn)
    return dataloader