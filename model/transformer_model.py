import json
from tqdm import tqdm  # 引入tqdm
import re
import os
import torch
from configs import cfgs
import torch.nn as nn
import math  # 确保导入math库
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
def load_and_merge_json_files(filenames):
    merged_data = {'annotations': [], 'images': [], 'categories': []}
    for filename in tqdm(filenames, desc="Loading and merging JSON files"):
        file_path = os.path.join(cfgs.DATASETS.root, 'calculus_dataset_com', filename)
        with open(file_path, 'r') as file:
            data = json.load(file)
            merged_data['annotations'].extend(data.get('annotations', []))
            merged_data['images'].extend(data.get('images', []))
            if not merged_data['categories']:
                merged_data['categories'] = data.get('categories', [])
    return merged_data
def tokenize_latex(latex_string):
    # 识别LaTeX命令、括号、数字以及其他符号
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
    # 使用defaultdict创建一个以image_id为键，注释列表为值的字典
    annotation_dict = defaultdict(list)
    for annotation in annotations:
        annotation_dict[annotation['image_id']].append(annotation)
    # 对于每个image_id的注释列表，根据id排序（一次性操作）
    for ann_list in annotation_dict.values():
        ann_list.sort(key=lambda x: x['id'])
    preprocessed_data = []
    # 定义处理每个图像的函数
    def process_image(image):
        image_id = image['id']
        image_width = image['width']
        image_height = image['height']
        latex_string = image['latex']
        latex_tokens = tokenize_latex(latex_string)
        image_annotations = annotation_dict[image_id]
        token_bboxes = []
        for token, annotation in zip(latex_tokens, image_annotations):
            bbox = annotation['bbox']
            normalized_bbox = [
                bbox[0] / image_width,
                bbox[1] / image_height,
                bbox[2] / image_width,
                bbox[3] / image_height
            ]
            token_bboxes.append((token, normalized_bbox))
        return token_bboxes
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_image, images), total=len(images), desc="Preprocessing data"))
    preprocessed_data.extend(results)
    return preprocessed_data
# JSON文件列表
filenames = [
    'calculus_train.json',
    'calculus_test.json',
    'calculus_val.json'
]
merged_data = load_and_merge_json_files(filenames)
# 进行预处理
preprocessed_data = preprocess_data(merged_data)
class TransformerModel(nn.Module):
    def __init__(self, num_tokens, num_special_tokens, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder_embeddings = nn.Embedding(num_tokens, d_model)
        self.num_special_tokens = num_special_tokens
        self.bbox_embeddings = nn.Linear(4, d_model)  # 将bbox的4个坐标映射到d_model维
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
        self.out = nn.Linear(d_model, num_tokens)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, device=self.device)) == 1
        mask = mask.transpose(0, 1).float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def forward(self, src, tgt, bboxes, src_mask=None):
        src_embeddings = self.encoder_embeddings(src) + self.bbox_embeddings(bboxes)
        src_embeddings = self.pos_encoder(src_embeddings)
        if src_mask is None or src_mask.size(0) != len(src):
            src_mask = self.generate_square_subsequent_mask(len(src)).to(src.device)
        encoder_output = self.transformer_encoder(src_embeddings, src_mask)
        tgt_embeddings = self.encoder_embeddings(tgt)
        tgt_embeddings = self.pos_encoder(tgt_embeddings)
        tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(tgt.device)
        output = self.transformer_decoder(tgt_embeddings, encoder_output, tgt_mask)
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
print(train_token_to_id)
test_token_to_id = create_token_to_id_mapping(test_data)
print(test_token_to_id)
train_dataset = LaTeXDataset(train_data, train_token_to_id)
test_dataset = LaTeXDataset(test_data, test_token_to_id)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_token_id = max([max(ids) for ids, _ in train_dataset])
total_tokens = max_token_id + 1
num_special_tokens = 10
model = TransformerModel(num_tokens=total_tokens + num_special_tokens, num_special_tokens=num_special_tokens, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
model = model.to(device)
min_token_id = min([min(ids) for ids, _ in train_dataset])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Training', leave=False)  # 创建进度条
    for token_ids, bboxes in progress_bar:
        print(f'token_ids shape: {token_ids.shape}, bboxes shape: {bboxes.shape}')
        token_ids, bboxes = token_ids.to(device), bboxes.to(device)
        optimizer.zero_grad()
        output = model(token_ids, token_ids, bboxes)  # 确保传递bbox信息
        loss = criterion(output.transpose(1, 2), token_ids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'Loss': loss.item()})  # 更新进度条
    return total_loss / len(data_loader)
# 训练模型
epochs = 10
def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Validation', leave=False)  # 创建进度条
    with torch.no_grad():
        for token_ids, bboxes in progress_bar:
            token_ids, bboxes = token_ids.to(device), bboxes.to(device)
            output = model(token_ids, token_ids, bboxes)  # 应保持与train一致
            loss = criterion(output.transpose(1, 2), token_ids)
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})  # 更新进度条
    return total_loss / len(data_loader)
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, test_loader, criterion, device)
    print(f'Epoch {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}')
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        # Save the model with validation loss in the filename
        filename = f'best_model_epoch_{epoch}_val_loss_{val_loss:.4f}.pth'  # Format loss to 4 decimal places
        file_path = os.path.join(cfgs.CHECKPOINTS.root, 'com', filename)
        torch.save(model.state_dict(), file_path)
        print(f'Model saved as {filename} - Epoch {epoch+1}, Validation Loss: {val_loss}')

