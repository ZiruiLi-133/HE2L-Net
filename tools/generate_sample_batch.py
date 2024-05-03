from utils import get_calc_dataloader
import os
import numpy
import json
from configs import cfgs

if __name__ == '__main__':
    dataloader = get_calc_dataloader(batch_size=1)
    output_json_file_path = os.path.join(cfgs.OUTPUTS.root, 'sample_data_batch.json')
    sample_batch = {}
    data_iter = iter(dataloader)
    batch = next(data_iter)
    print(batch)
    sample_batch['image'] = batch['image'].numpy().tolist()
    sample_batch['image_size'] = batch['image_size']
    sample_batch['latex_code'] = batch['latex_code']
    sample_batch['boxes'] = batch['boxes'].numpy().tolist()
    sample_batch['labels'] = batch['labels'].numpy().tolist()
    sample_batch['labels_str'] = batch['labels_str']
    
    with open(output_json_file_path, 'w') as f:
        json.dump(sample_batch, f, indent=4)