import os
from ultralytics import YOLO
from configs import cfgs
import torch

if __name__ == '__main__':
    model = YOLO('yolov8x.yaml')
    model = model.load(os.path.join(cfgs.CHECKPOINTS.root, 'yolov8x.pt'))
    results = model.train(data=cfgs.TRAIN_Rec.DATASET.config, epochs=2, imgsz=640)