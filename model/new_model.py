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


def create_token_to_id_mapping():
    # token_to_id = defaultdict(lambda: len(token_to_id))
    # for sample in preprocessed_data:
    #     for token, _ in sample:
    #         _ = token_to_id[token]
    # # print(f'total number of tokens: {len(token_to_id)}')
    dict = None
    with open(os.path.join(cfgs.TRAIN_Com.DATASET.root, 'symbol_to_id.json'), 'r') as f:
        dict = json.load(f)
    return dict

def create_id_to_token_mapping():
    id_to_symbol = {}
    file_path = os.path.join(cfgs.TRAIN_Com.DATASET.root, 'id_to_symbol.json')
    with open(file_path, 'r') as f:
        dict_str_keys = json.load(f)
        id_to_symbol = {int(k): v for k, v in dict_str_keys.items()}
    # print(f'id_to_symbol: {id_to_symbol}')
    return id_to_symbol

global_token_to_id = create_token_to_id_mapping()
global_id_to_token = create_id_to_token_mapping()


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
    ground_truth_latex = []

    def process_image(image):
        image_id = image['id']
        image_width = image['width']
        image_height = image['height']
        full_sequence = image['full_latex_chars']
        image_annotations = annotation_dict[image_id]

        ground_truth_latex.append(full_sequence)
        
        # 将每个 LaTeX token 与注释中的 bbox 关联
        token_bboxes = []
        for annotation in image_annotations:
            bbox = annotation['bbox']
            normalized_bbox = [
                bbox[0] / image_width,
                bbox[1] / image_height,
                bbox[2] / image_width,
                bbox[3] / image_height
            ]
            token_bboxes.append((annotation['category_id'], normalized_bbox))

        return token_bboxes

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_image, images), total=len(images), desc="Preprocessing data"))
    preprocessed_data.extend(results)

    return preprocessed_data, ground_truth_latex


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
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
        self.out = nn.Linear(d_model, num_tokens)
        self.id_to_token = global_id_to_token
        
    def tensor_to_latex(self, output):
        # Get the indices with the highest probability, which are our predicted class indices
        predicted_indices = torch.argmax(output, dim=-1)
        
        # Convert these indices to symbols using the id_to_symbol dictionary
        batch_size = output.size(0)
        result = []
        for i in range(batch_size):
            sequence = [self.id_to_token[index.item()] for index in predicted_indices[i]]
            # Join the symbols into a single string
            latex_string = ''.join(sequence)
            result.append(latex_string)
            
        return result
    
    def id_to_latex(self, batch):
        batch_size = batch.size(0)
        result = []
        for i in range(batch_size):
            sequence = [self.id_to_token[index.item()] for index in batch[i]]
            # Join the symbols into a single string
            latex_string = ''.join(sequence)
            result.append(latex_string)
        return result
        
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, device=self.device)) == 1
        mask = mask.transpose(0, 1).float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate_padding_mask(self, input_tensor, pad_idx=global_token_to_id['<pad>']):
        """
        Generate a padding mask where positions with a specific pad_idx are marked as `True`.
        
        Args:
            input_tensor (torch.Tensor): The input tensor with shape `[batch_size, seq_length]`.
            pad_idx (int): The index representing the padding tokens.
        
        Returns:
            torch.Tensor: A mask tensor of shape `[batch_size, seq_length]`.
        """
        return (input_tensor == pad_idx)
        
    def shift_and_prepend_start_token(self, gt_sequences):
        batch_size = gt_sequences.size(0)
        
        # Create an empty tensor to hold the new target sequences
        new_gt_sequences = torch.full((batch_size, gt_sequences.size(1)), global_token_to_id['<pad>'], dtype=torch.long)
        
        # Prepend the <start> token and shift the rest
        new_gt_sequences[:, 0] = global_token_to_id['<start>']
        new_gt_sequences[:, 1:] = gt_sequences[:, :-1]

        return new_gt_sequences
    
    def forward(self, src, bboxes, gt=None, visualize=False, is_training=False, src_mask=None):
        # print(f'src shape: {src.shape}')
        src_embeddings = self.encoder_embeddings(src) + self.bbox_embeddings(bboxes)
        src_embeddings = self.pos_encoder(src_embeddings)
        # print(f'src_embeddings shape: {src_embeddings.shape}')
        if src_mask is None or src_mask.size(0) != len(src):
            src_padding_mask = self.generate_padding_mask(src).to(src.device)
        # print(f'src_padding_mask shape: {src_padding_mask.shape}')
        # print(src_padding_mask)
        encoder_output = self.transformer_encoder(src_embeddings, src_key_padding_mask=src_padding_mask)
        # print(f'encoder_output shape: {encoder_output.shape}')
        if is_training:
            tgt = self.shift_and_prepend_start_token(gt).to(src.device)
        else:
            tgt = torch.full((src.size(0), 1), global_token_to_id['<start>'], dtype=torch.long).to(src.device)
            
        # print(f'tgt shape: {tgt.shape}')
        tgt_embeddings = self.encoder_embeddings(tgt)
        tgt_embeddings = self.pos_encoder(tgt_embeddings)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        # print(f'tgt_mask shape: {tgt_mask.shape}')
        output = self.transformer_decoder(tgt_embeddings, encoder_output, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
        output = self.out(output)

        # print(f'output shape: {output.shape}')
        # print(self.tensor_to_latex(output))
        if visualize:
            print('predicted:')
            print(self.tensor_to_latex(output)[0])
            print('gt:')
            print(self.id_to_latex(gt)[0])
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

class LaTeXDataset(Dataset):
    
    def __init__(self, data, gt):
        self.data = data
        self.gt = gt
        self.token_to_id = global_token_to_id
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        token_ids, bboxes = zip(*self.data[idx])
        gt_seq = self.gt[idx]
        gt_ids = [self.token_to_id[gt] for gt in gt_seq]
        # token_ids = [self.token_to_id[token] for token in tokens]
        bboxes = torch.tensor(bboxes, dtype=torch.float32)  # 确保bbox是float类型
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        gt_ids = torch.tensor(gt_ids, dtype=torch.long)
        return token_ids, bboxes, gt_ids
    
def collate_fn(batch):
    token_ids, bboxes, gt = zip(*batch)
    token_ids_padded = pad_sequence(token_ids, batch_first=True, padding_value=global_token_to_id['<pad>'])
    bboxes_padded = pad_sequence(bboxes, batch_first=True, padding_value=global_token_to_id['<pad>'])
    gt_padded = pad_sequence(gt, batch_first=True, padding_value=global_token_to_id['<pad>'])
    return token_ids_padded, bboxes_padded, gt_padded

def pad_or_truncate_output(output, target_length, padding_index=global_token_to_id['<pad>']):
    """
    Pads or truncates the output sequences to match the target length.
    For padding, it sets the class index `padding_index` to 1 and other classes to 0.
    
    Args:
        output (torch.Tensor): Model output tensor with shape `[batch_size, seq_length, num_classes]`.
        target_length (int): Target sequence length for padding/truncating.
        padding_index (int, optional): Index to use for padding. Defaults to 0.
    
    Returns:
        torch.Tensor: The padded or truncated output tensor.
    """
    batch_size, seq_length, num_classes = output.shape

    # If the model output is shorter than the target length, pad the extra positions
    if seq_length < target_length:
        # Create a tensor with all zeros initially
        padding = torch.zeros((batch_size, target_length - seq_length, num_classes), device=output.device)

        # Set the probability of the padding class (padding_index) to 1
        # print(f"padding_index: {padding_index}")
        padding[:, :, padding_index] = 1.0

        # Concatenate the original output with the padding
        output_padded = torch.cat([output, padding], dim=1)

    # If the model output is longer than the target length, truncate it
    elif seq_length > target_length:
        output_padded = output[:, :target_length, :]

    else:  # If they are the same length, no modification is required
        output_padded = output

    return output_padded

def append_end_token(gt_sequences, end_token_idx=global_token_to_id['<end>'], pad_token_idx=global_token_to_id['<pad>']):
    """
    Append an <end> token to each sequence in the batch and pad them to the same length.
    
    Args:
        gt_sequences (torch.Tensor): Ground truth tensor of shape `[batch_size, seq_length]`.
        end_token_idx (int): Index of the <end> token.
        pad_token_idx (int): Index of the <pad> token.
    
    Returns:
        torch.Tensor: New tensor with the <end> token appended and sequences padded.
    """
    batch_size, seq_length = gt_sequences.shape
    new_length = seq_length + 1

    # Initialize a new tensor with the additional length and fill with the <pad> token
    padded_sequences = torch.full((batch_size, new_length), pad_token_idx, dtype=gt_sequences.dtype,
                                  device=gt_sequences.device)

    # Copy the original sequences and append the <end> token at the new final position
    padded_sequences[:, :seq_length] = gt_sequences
    padded_sequences[:, seq_length] = end_token_idx

    return padded_sequences



def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Training', leave=False)  # 创建进度条
    for token_ids, bboxes, gt_ids in progress_bar:
        token_ids, bboxes, gt_ids = token_ids.to(device), bboxes.to(device), gt_ids.to(device)
        gt_end = append_end_token(gt_ids)
        optimizer.zero_grad()
        output = model(token_ids, bboxes, gt_ids, visualize=True, is_training=True)  # 确保传递bbox信息
        # print(f'gt_ids shape {gt_ids.shape}')
        max_length = gt_end.shape[1]
        output = pad_or_truncate_output(output, max_length)
        # print(f'padded/truncated output shape: {output.shape}')
        # print(f'gt_end shape: {gt_end.shape}')
        loss = criterion(output.transpose(1, 2), gt_end)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'Loss': loss.item()})
    return total_loss / len(data_loader)

def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    progress_bar = tqdm(data_loader, desc='Validation', leave=False)  # 创建进度条
    with torch.no_grad():
        for token_ids, bboxes, gt_ids in progress_bar:
            token_ids, bboxes, gt_ids = token_ids.to(device), bboxes.to(device), gt_ids.to(device)
            gt_end = append_end_token(gt_ids)
            output = model(token_ids, bboxes, gt_ids, visualize=True, is_training=False)
            max_length = gt_end.shape[1]
            output = pad_or_truncate_output(output, max_length)
            loss = criterion(output.transpose(1, 2), gt_end)
            total_loss += loss.item()
            # 保存预测和真实标签用于计算精确率、召回率、F1分数
            preds = torch.argmax(output, dim=-1)
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_targets.extend(gt_end.view(-1).cpu().numpy())

            progress_bar.set_postfix({'Loss': loss.item()})  # 更新进度条


    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, labels=list(range(1, max_token_id+1)), average='macro', zero_division=0)
    return total_loss / len(data_loader), precision, recall, f1

if __name__ == '__main__':
    with open(os.path.join(cfgs.DATASETS.root, 'calculus_dataset_com', 'calculus_train.json'), 'r') as f:
        train_data = json.load(f)
    with open(os.path.join(cfgs.DATASETS.root, 'calculus_dataset_com', 'calculus_val.json'), 'r') as f:
        val_data = json.load(f)
    preprocessed_train_data, train_ground_truth = preprocess_single_json(train_data)
    preprocessed_val_data, val_ground_truth = preprocess_single_json(val_data)

    train_dataset = LaTeXDataset(preprocessed_train_data, train_ground_truth)
    val_dataset = LaTeXDataset(preprocessed_val_data, val_ground_truth)
    batch_size = cfgs.TRAIN_Com.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_token_id = max([max(ids) for ids, _, _ in train_dataset])
    total_tokens = 63
    num_special_tokens = 0
    model = TransformerModel(num_tokens=total_tokens + num_special_tokens, num_special_tokens=num_special_tokens, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
    model = model.to(device)
    if not cfgs.TRAIN_Com.start_new:
        checkpoint_path = os.path.join(cfgs.CHECKPOINTS.root, 'com', 'best.pth')
        checkpoint = torch.load(checkpoint_path, map_location=device)  # Load checkpoint
        model.load_state_dict(checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0000001)
    criterion = nn.CrossEntropyLoss()

    epochs = cfgs.TRAIN_Com.epochs
    highest_precision = 0
    highest_recall = 0
    highest_f1 = 0
    val_loss, precision, recall, f1 = validate(model, val_dataloader, criterion, device)
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, precision, recall, f1 = validate(model, val_dataloader, criterion, device)
        
        print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        if precision > highest_precision:
            # file_name = f'epoch_{epoch}_precision_{precision}_recall_{recall}_f1_{f1}.pth'
            file_path = os.path.join(cfgs.CHECKPOINTS.root, 'com', 'best.pth')
            torch.save(model.state_dict(), file_path)
            highest_precision = precision
        if recall > highest_recall:
            highest_recall = recall
        if f1 > highest_f1:
            highest_f1 = f1
        print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        print(f'Highest metrics: {highest_precision}, {highest_recall}, {highest_f1}')