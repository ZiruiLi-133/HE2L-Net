import json
from tqdm import tqdm 
import re
import os
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from configs import cfgs
import torch
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

def tokenize_latex_from_chars(full_latex_chars):
    """
    使用full_latex_chars来返回token
    """
    return full_latex_chars

def preprocess_single_json(data):
    images = data['images']
    annotations = data['annotations']
    categories = {category['id']: category['name'] for category in data['categories']}

    # 通过 defaultdict 创建一个以 image_id 为键，注释列表为值的字典
    annotation_dict = defaultdict(list)
    for annotation in annotations:
        annotation_dict[annotation['image_id']].append(annotation)

    # 为每个 image_id 排序注释列表（以 id 为关键字）
    for ann_list in annotation_dict.values():
        ann_list.sort(key=lambda x: x['id'])

    preprocessed_data = []

    def process_image(image):
        image_id = image['id']
        image_width = image['width']
        image_height = image['height']
        latex_tokens = tokenize_latex_from_chars(image['full_latex_chars'])
        image_annotations = annotation_dict[image_id]

        # 将每个 LaTeX token 与注释中的 bbox 关联
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
        print(f'src shape: {src.shape}')
        src_embeddings = self.encoder_embeddings(src) + self.bbox_embeddings(bboxes)
        src_embeddings = self.pos_encoder(src_embeddings)
        print(f'src_embedding shape: {src_embeddings.shape}')
        if src_mask is None or src_mask.size(0) != len(src):
            src_mask = self.generate_square_subsequent_mask(len(src)).to(src.device)
        print(f'src_mask shape: {src_mask.shape}')
        encoder_output = self.transformer_encoder(src_embeddings, src_mask)
        print(f'encoder_output shape: {encoder_output.shape}')
        tgt_embeddings = self.encoder_embeddings(tgt)
        tgt_embeddings = self.pos_encoder(tgt_embeddings)
        tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(tgt.device)
        output = self.transformer_decoder(tgt_embeddings, encoder_output, tgt_mask)
        output = self.out(output)
        print(f'output shape: {output.shape}')
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
    # print(f'total number of tokens: {len(token_to_id)}')
    return dict(token_to_id)

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

def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Training', leave=False)  # 创建进度条
    for token_ids, bboxes in progress_bar:
        token_ids, bboxes = token_ids.to(device), bboxes.to(device)
        optimizer.zero_grad()
        output = model(token_ids, token_ids, bboxes)  # 确保传递bbox信息
        print(f"output: {output}")
        loss = criterion(output.transpose(1, 2), token_ids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'Loss': loss.item()})  # 更新进度条
    return total_loss / len(data_loader)

def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    progress_bar = tqdm(data_loader, desc='Validation', leave=False)  # 创建进度条
    with torch.no_grad():
        for token_ids, bboxes in progress_bar:
            token_ids, bboxes = token_ids.to(device), bboxes.to(device)
            output = model(token_ids, token_ids, bboxes)  # 应保持与train一致
            loss = criterion(output.transpose(1, 2), token_ids)
            total_loss += loss.item()

            # 保存预测和真实标签用于计算精确率、召回率、F1分数
            preds = torch.argmax(output, dim=-1)
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_targets.extend(token_ids.view(-1).cpu().numpy())

            progress_bar.set_postfix({'Loss': loss.item()})  # 更新进度条


    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, labels=list(range(1, max_token_id+1)), average='macro', zero_division=0)
    return total_loss / len(data_loader), precision, recall, f1

if __name__ == '__main__':
    with open(os.path.join(cfgs.DATASETS.root, 'calculus_dataset_com', 'calculus_train.json'), 'r') as f:
        train_data = json.load(f)
    with open(os.path.join(cfgs.DATASETS.root, 'calculus_dataset_com', 'calculus_val.json'), 'r') as f:
        val_data = json.load(f)

    preprocessed_train_data = preprocess_single_json(train_data)
    preprocessed_val_data = preprocess_single_json(val_data)

    train_token_to_id = create_token_to_id_mapping(preprocessed_train_data)
    print(train_token_to_id)
    val_token_to_id = create_token_to_id_mapping(preprocessed_val_data)
    train_dataset = LaTeXDataset(preprocessed_train_data, train_token_to_id)
    val_dataset = LaTeXDataset(preprocessed_val_data, val_token_to_id)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_token_id = max([max(ids) for ids, _ in train_dataset])
    total_tokens = max_token_id + 1
    num_special_tokens = 10
    model = TransformerModel(num_tokens=total_tokens + num_special_tokens, num_special_tokens=num_special_tokens, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
    model = model.to(device)
    min_token_id = min([min(ids) for ids, _ in train_dataset])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, precision, recall, f1 = validate(model, val_dataloader, criterion, device)
        print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        