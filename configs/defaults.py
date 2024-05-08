import os
from yacs.config import CfgNode as CN

# Access the environment variable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

_C = CN()
_C.PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

# ----Configs----
_C.CONFIGS = CN()
_C.CONFIGS.root = os.path.join(_C.PROJECT_ROOT, 'configs')
_C.CONFIGS.dataset_configs = os.path.join(_C.CONFIGS.root, 'datasets')

# ----weights----
_C.WEIGHTS = CN()
_C.CONFIGS.root = os.path.join(_C.PROJECT_ROOT, 'weights')

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
_C.TRAIN_Rec.start_new = True
_C.TRAIN_Rec.batch_size = 1
_C.TRAIN_Rec.epochs = 40
_C.TRAIN_Rec.DATASET = CN()
_C.TRAIN_Rec.DATASET.root = os.path.join(_C.DATASETS.root, 'calculus_dataset_yolo')
_C.TRAIN_Rec.DATASET.train_img = os.path.join(_C.TRAIN_Rec.DATASET.root, 'train', 'images')
_C.TRAIN_Rec.DATASET.test_img = os.path.join(_C.TRAIN_Rec.DATASET.root, 'test', 'images')
_C.TRAIN_Rec.DATASET.config = os.path.join(_C.CONFIGS.dataset_configs, 'calculus_dataset_yolo.yaml')
_C.TRAIN_Rec.image_size = 640

# ----Train Composer----
_C.TRAIN_Com = CN()
_C.TRAIN_Com.DATASET = CN()
_C.TRAIN_Com.batch_size = 32    
_C.TRAIN_Com.epochs = 1000
_C.TRAIN_Com.start_new = True
_C.TRAIN_Com.normalize_bbox = True
_C.TRAIN_Com.DATASET.root = os.path.join(_C.DATASETS.root, 'calculus_dataset_com')
_C.TRAIN_Com.DATASET.image_root = os.path.join(_C.TRAIN_Com.DATASET.root, 'calculus_images')