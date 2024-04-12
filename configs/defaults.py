import os
from yacs.config import CfgNode as CN

# Access the environment variable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

_C = CN()
_C.PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

# ----Dataset----
_C.DATASETS = CN()
_C.DATASETS.root = os.path.join(_C.PROJECT_ROOT, 'datasets')

# ----Outputs----
_C.OUTPUTS = CN()
_C.OUTPUTS.root = os.path.join(_C.PROJECT_ROOT, 'outputs')

# ----Checkpoints----
_C.CHECKPOINTS = CN()
_C.CHECKPOINTS.root = os.path.join(_C.OUTPUTS.root, 'checkpoints')

# ----Train YOLO----
_C.TRAIN_Rec = CN()
_C.TRAIN_Rec.model = 'YOLO'
_C.TRAIN_Rec.batch_size = 1
_C.TRAIN_Rec.epochs = 2
_C.TRAIN_Rec.DATASET = CN()
_C.TRAIN_Rec.DATASET.root = os.path.join(_C.DATASETS.root, 'kaggle_data_coco')
_C.TRAIN_Rec.DATASET.config = os.path.join(_C.TRAIN_Rec.DATASET.root, 'kaggle_data_coco.yaml')
_C.TRAIN_Rec.image_size = 640