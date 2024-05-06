import json
from tqdm import tqdm 
import re
import pickle
import os
import torch
import torch.nn as nn
import math  # 确保导入math库
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from sklearn.model_selection import train_test_split
from collections import defaultdict
from configs import cfgs
def load_and_merge_json_files(filenames):
    
    merged_data = {'annotations': [], 'images': [], 'categories': []}
    for filename in tqdm(filenames, desc="Loading and merging JSON files"):
        file_path = os.path.join(cfgs.DATASETS.root, 'calculus_dataset_com', filename)
        with open(file_path, 'r') as file:
            data = json.load(file)
            merged_data['annotations'].extend(data.get('annotations', []))
            merged_data['images'].extend(data.get('images', []))
            # 假设categories在所有文件中都是相同的，所以只需要从一个文件读取
            if not merged_data['categories']:
                merged_data['categories'] = data.get('categories', [])
    return merged_data

def tokenize_latex(latex_string):
    tokens = re.findall(r'\\[a-zA-Z]+|\{|\}|\[|\]|\(|\)|\d+|\S', latex_string)
    balance = 0
    for token in tokens:
        if token in ['{', '[', '(']:
            balance += 1
        elif token in ['}', ']', ')']:
            balance -= 1
        if balance < 0:
            raise ValueError("Unbalanced brackets detected in the LaTeX string.")
    if balance != 0:
        raise ValueError("Unbalanced brackets detected in the LaTeX string.")
    return tokens

def preprocess_data(data):
    images = data['images']
    annotations = data['annotations']
    categories = {category['id']: category['name'] for category in data['categories']}
    preprocessed_data = []
    for image in tqdm(images, desc="Preprocessing data"):
        image_id = image['id']
        image_width = image['width']
        image_height = image['height']
        latex_string = image['latex']
        latex_tokens = tokenize_latex(latex_string)
        image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]
        image_annotations.sort(key=lambda x: x['id'])  # 确保顺序与tokens一致
        token_bboxes = []
        for token, annotation in zip(latex_tokens,image_annotations ):
            bbox = annotation['bbox']
            normalized_bbox = [
                bbox[0] / image_width,
                bbox[1] / image_height,
                bbox[2] / image_width,
                bbox[3] / image_height
            ]
            token_bboxes.append((token, normalized_bbox))
        preprocessed_data.append(token_bboxes)
    
    output_file_path = os.path.join(cfgs.OUTPUTS.root, 'preprocessed_data.pkl')
    with open(output_file_path, 'wb') as file:
        pickle.dump(preprocessed_data, file)
    return preprocessed_data
# JSON文件列表
filenames = [
    'calculus_train.json',
    'calculus_val.json',
    'calculus_test.json'
]
merged_data = load_and_merge_json_files(filenames)
# 进行预处理
preprocessed_data = preprocess_data(merged_data)
print(preprocessed_data)

class TransformerModel(nn.Module):
    def __init__(self, num_tokens, num_special_tokens, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.encoder_embeddings = nn.Embedding(num_tokens, d_model)
        self.special_command_embeddings = nn.Embedding(num_special_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
        self.out = nn.Linear(d_model, num_tokens)
    def select_embeddings(self, tokens):
        normal_tokens_mask = tokens < self.encoder_embeddings.num_embeddings
        special_tokens_mask = ~normal_tokens_mask
        embeddings = torch.zeros(tokens.size(0), self.d_model, device=tokens.device)
        embeddings[normal_tokens_mask] = self.encoder_embeddings(tokens[normal_tokens_mask])
        embeddings[special_tokens_mask] = self.special_command_embeddings(tokens[special_tokens_mask])
        return embeddings * math.sqrt(self.d_model)
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, device=self.device)) == 1
        mask = mask.transpose(0, 1).float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def forward(self, src, tgt, src_mask=None):
        if src_mask is None or src_mask.size(0) != len(src):
            src_mask = self.generate_square_subsequent_mask(len(src)).to(src.device)

        src_embeddings = self.select_embeddings(src)
        src_embeddings = self.pos_encoder(src_embeddings)
        encoder_output = self.transformer_encoder(src_embeddings, src_mask)

        tgt_embeddings = self.select_embeddings(tgt)
        tgt_embeddings = self.pos_encoder(tgt_embeddings)
        output = self.transformer_decoder(tgt_embeddings, encoder_output)
        output = self.out(output)
        return output
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
def create_token_to_id_mapping(preprocessed_data):
    token_to_id = defaultdict(lambda: len(token_to_id))
    for sample in preprocessed_data:
        for token, _ in sample:
            _ = token_to_id[token]
    return dict(token_to_id)
token_to_id = create_token_to_id_mapping(preprocessed_data)
class LaTeXDataset(Dataset):
    def __init__(self, data, token_to_id):
        self.data = data
        self.token_to_id = token_to_id
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        tokens, bboxes = zip(*self.data[idx])
        token_ids = [self.token_to_id[token] for token in tokens]
        bboxes = torch.tensor(bboxes, dtype=torch.float32)  # 确保bbox是float类型
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        return token_ids, bboxes
def collate_fn(batch):
    token_ids, bboxes = zip(*batch)
    token_ids_padded = pad_sequence(token_ids, batch_first=True, padding_value=0)
    bboxes_padded = pad_sequence(bboxes, batch_first=True, padding_value=0)
    return token_ids_padded, bboxes_padded
train_data, test_data = train_test_split(preprocessed_data, test_size=0.2, random_state=42)
train_token_to_id = create_token_to_id_mapping(train_data)
test_token_to_id = create_token_to_id_mapping(test_data)
train_dataset = LaTeXDataset(train_data, train_token_to_id)
test_dataset = LaTeXDataset(test_data, test_token_to_id)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(num_tokens=100, num_special_tokens=10)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for token_ids, bboxes in data_loader:
        token_ids, bboxes = token_ids.to(device), bboxes.to(device)
        optimizer.zero_grad()
        output = model(token_ids, bboxes)
        loss = criterion(output.transpose(1, 2), token_ids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)
# 训练模型
epochs = 10
def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for token_ids, bboxes in data_loader:
            token_ids, bboxes = token_ids.to(device), bboxes.to(device)
            output = model(token_ids, token_ids)
            loss = criterion(output.transpose(1, 2), token_ids)
            total_loss += loss.item()
    return total_loss / len(data_loader)
min_val_loss = float('inf')
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, test_loader, criterion, device)
    print(f'Epoch {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}')
    torch.save(model.state_dict(), 'best_model.pth')
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        # Save the model with validation loss in the filename
        filename = f'best_model_epoch_{epoch}_val_loss_{val_loss:.4f}.pth'  # Format loss to 4 decimal places
        file_path = os.path.join(cfgs.CHECKPOINTS.root, 'com', filename)
        torch.save(model.state_dict(), file_path)
        print(f'Model saved as {filename} - Epoch {epoch+1}, Validation Loss: {val_loss}')
