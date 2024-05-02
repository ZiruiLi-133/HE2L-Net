import os
from configs import cfgs
import json

symbol_to_id = {}
id_to_symbol = {}
id_counter = 0
json_file = 'calculus_train.json'
json_file_path = os.path.join(cfgs.TRAIN_Com.DATASET.root, json_file)
annotations = None

additional_symbols = ["^", "{", "}"]

with open(json_file_path) as f:
    annotations = json.load(f)
categories = annotations['categories']
for dict in categories:
    id = dict['id']
    name = dict['name']
    symbol_to_id[name] = id
    id_to_symbol[id] = name
    id_counter += 1

for symbol in additional_symbols:
    symbol_to_id[symbol] = id_counter
    id_to_symbol[id_counter] = symbol
    id_counter += 1

with open('symbol_to_id.json', 'w') as json_file:
    json.dump(symbol_to_id, json_file, indent=4)

with open('id_to_symbol.json', 'w') as json_file:
    json.dump(id_to_symbol, json_file, indent=4)