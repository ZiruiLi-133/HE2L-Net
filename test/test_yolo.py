from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    model = YOLO('yolov8n.yaml')
    model = model.load('yolov8n.pt')
    results = model.train(data='coco128.yaml', epochs=2, imgsz=640)