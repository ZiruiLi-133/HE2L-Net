import os
from configs import cfgs
import json

symbol_to_id = {}
id_to_symbol = {}
id_counter = 0
json_file = 'calculus_train.json'
json_file_path = os.path.join(cfgs.TRAIN_Com.DATASET.root, json_file)
annotations = None

with open(json_file_path) as f:
    annotations = json.load(f)
categories = annotations['categories']
for dict in categories:
    id = dict['id']
    print(id)
    name = dict['name']
    symbol_to_id[name] = id
    id_to_symbol[id] = name
    id_counter += 1

    
for image in annotations['images']:
    for symbol in image['full_latex_chars']:
        if symbol not in symbol_to_id:
            symbol_to_id[symbol] = id_counter
            id_to_symbol[id_counter] = symbol  # Make sure the keys are strings if your JSON uses string keys
            id_counter += 1


special_tokens = ['<start>', '<end>', '<pad>']
for special_token in special_tokens:
    symbol_to_id[special_token] = id_counter
    id_to_symbol[id_counter] = special_token
    id_counter += 1

with open(os.path.join(cfgs.TRAIN_Com.DATASET.root, 'symbol_to_id.json'), 'w') as json_file:
    json.dump(symbol_to_id, json_file, indent=4)

with open(os.path.join(cfgs.TRAIN_Com.DATASET.root, 'id_to_symbol.json'), 'w') as json_file:
    json.dump(id_to_symbol, json_file, indent=4)