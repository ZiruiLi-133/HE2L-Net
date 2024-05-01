import json
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

# Assume latex_tokens is correctly populated with the unique LaTeX commands as strings
latex_tokens = ["\\lim_", "a", "\\to", "\\frac", "\\pi", "4", "d", "\\left(", "\\sin", "+", "-", "6", "\\sec", "\\right)", "w", "/",
                "5", "\\tan", "2", "3", "e", "b", "7", "\\cos", "\\theta", "8", "=", "x", "9", "1", "y", "h", "k", "g", "\\csc", 
                "\\infty", "0", "\\sqrt", "r", "\\ln", "n", "u", "\\cot", "\\left|", "\\right|", "p", "t", "z", "\\log", "v", "s", "c", "\\cdot", "."]

# Image Encoder Class
class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim, pretrained=True):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, hidden_dim))
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        features = self.adaptive_pool(features)
        features = features.squeeze(-2).permute(0, 2, 1)
        _, (hidden, cell) = self.rnn(features)
        return hidden, cell

# Decoder Class
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

# Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(src)
        
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            top1 = output.argmax(1)
            
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else top1
        
        return outputs
    
    
# Dataset and DataLoader
class MathExpressionDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        with open(json_file) as f:
            self.annotations = json.load(f)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        img_id = self.annotations['images'][idx]['id']
        img_name = os.path.join(self.img_dir, self.annotations['images'][idx]['file_name'])
        image = Image.open(img_name).convert('RGB')

        annotations = [a for a in self.annotations['annotations'] if a['image_id'] == img_id]
        boxes = torch.as_tensor([a['bbox'] for a in annotations], dtype=torch.float32)
        labels = torch.as_tensor([a['category_id'] for a in annotations], dtype=torch.int64)
        labels_str = [self.annotations['categories'][label.item() - 1]['name'] for label in labels]
        
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'boxes': boxes, 'labels': labels, 'labels_str': labels_str}
        return sample

def custom_collate_fn(batch):
    images = [item['image'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    labels_str = [item['labels_str'] for item in batch]

    collated_images = default_collate(images)
    collated_boxes = default_collate(boxes)
    collated_labels = default_collate(labels)

    return {
        'image': collated_images,
        'boxes': collated_boxes,
        'labels': collated_labels,
        'labels_str': labels_str
    }

def get_calc_dataloader(batch_size, shuffle=True, num_workers=1):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Add this line to resize images to a consistent size
        transforms.ToTensor(),
    ])

    dataset = MathExpressionDataset(json_file='G:\\ECE_208\\HE2L-Net\\datasets\\kaggle_data_coco\\kaggle_data_coco.json',
                                    img_dir='G:\\ECE_208\\HE2L-Net\\datasets\\archive\\batch_1\\background_images',
                                    transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=custom_collate_fn)
    return dataloader


# Training loop and data handling need to be implemented accordingly
def train(model, dataloader, optimizer, criterion, device):
    model.train()  # Set model to training mode
    total_loss = 0
    
    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)  # Assuming labels are the target LaTeX token indices

        optimizer.zero_grad()
        output = model(images, labels)  # Forward pass: Compute predicted output by passing images to the model
        
        # Calculate loss based on the output and the labels
        loss = criterion(output.view(-1, output_dim), labels.view(-1))
        loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()  # Perform a single optimization step (parameter update)
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

### Evaluation/Testing Loop
def evaluate(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    
    with torch.no_grad():  # Operations inside don't track history
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            output = model(images, labels)
            loss = criterion(output.view(-1, output_dim), labels.view(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


# Initialization of model, dataloader, optimizer, and loss
hidden_dim = 256
output_dim = len(latex_tokens)  # Including EOS/PAD might require adjusting the IDs in the dataset labels
embed_size = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = ImageEncoder(hidden_dim)
decoder = Decoder(output_dim, embed_size, hidden_dim)
seq2seq_model = Seq2Seq(encoder, decoder, device).to(device)
optimizer = optim.Adam(seq2seq_model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=output_dim-1)

train_dataloader = get_calc_dataloader(batch_size=10)  # Assuming this function returns a properly initialized DataLoader
test_dataloader = get_calc_dataloader(batch_size=10)  # This should ideally be different or separate data

# Example Main Training Loop
def main():
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(seq2seq_model, train_dataloader, optimizer, criterion, device)
        test_loss = evaluate(seq2seq_model, test_dataloader, criterion, device)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

if __name__ == '__main__':
    main()