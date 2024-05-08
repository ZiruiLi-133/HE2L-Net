import json
import os
import sys

from configs import cfgs

# Load existing id_to_symbol from JSON
id_to_symbol_path = os.path.join(cfgs.TRAIN_Com.DATASET.root, 'id_to_symbol.json')
with open(id_to_symbol_path, 'r') as file:
    id_to_symbol = json.load(file)

# Invert id_to_symbol to create symbol_to_id
symbol_to_id = {v: int(k) for k, v in id_to_symbol.items()}

# Load calculus_test.json
test_json_path = os.path.join(cfgs.TRAIN_Com.DATASET.root, 'calculus_test.json')
with open(test_json_path, 'r') as file:
    test_data = json.load(file)

# Track the maximum ID to assign new IDs to new symbols
max_id = max(symbol_to_id.values())

# Iterate through each image's full_latex_chars and update dictionaries
for image in test_data['images']:
    for symbol in image['full_latex_chars']:
        if symbol not in symbol_to_id:
            max_id += 1
            symbol_to_id[symbol] = max_id
            id_to_symbol[max_id] = symbol  # Make sure the keys are strings if your JSON uses string keys

special_tokens = ['<start>', '<end>', '<pad>']
for special_token in special_tokens:
    max_id += 1
    symbol_to_id[special_token] = max_id
    id_to_symbol[max_id] = special_token

# Save the updated symbol_to_id back to JSON
with open(os.path.join(cfgs.TRAIN_Com.DATASET.root, 'symbol_to_id.json'), 'w') as file:
    json.dump(symbol_to_id, file, indent=4)

# Save the updated id_to_symbol back to JSON
with open(os.path.join(cfgs.TRAIN_Com.DATASET.root, 'id_to_symbol.json'), 'w') as file:
    json.dump(id_to_symbol, file, indent=4)
