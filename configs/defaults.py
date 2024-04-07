import os
from yacs.config import CfgNode as CN

# Access the environment variable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

_C = CN()

# ----Dataset----
_C.DATASET = CN()
_C.DATASET.root = os.path.join(PROJECT_ROOT, 'datasets')

# ----Train YOLO----
_C.TRAIN_Rec = CN()
_C.TRAIN_Rec.model = 'YOLO'
_C.TRAIN_Rec.batch_size = 1
_C.TRAIN_Rec.epochs = 2