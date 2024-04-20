import os
import cv2
from ultralytics import YOLO
from configs import cfgs
import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':
    model = YOLO('yolov8x.yaml')
    # model = model.load(os.path.join(cfgs.CHECKPOINTS.root, 'yolov8x.pt'))
    model = model.load('/home/zli133/ECE_208/HE2L-Net/outputs/yolo_runs/train6/weights/last.pt')
    save_dir = os.path.join(cfgs.OUTPUTS.root, 'yolo_runs')
    # results = model.train(data=cfgs.TRAIN_Rec.DATASET.config, epochs=cfgs.TRAIN_Rec.epochs, imgsz=cfgs.TRAIN_Rec.image_size, project=save_dir, device=0)

    image_names = range(10)
    image_root = cfgs.TRAIN_Rec.DATASET.train_img
    image_path = [os.path.join(image_root, str(image_name)+'.jpg') for image_name in image_names]
    pred_result = model(image_path)
    # Visualization
    for i, result in enumerate(pred_result):
        rgb_img = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_img)
        plt.axis('off')
        print(result.boxes)

        for box in result.boxes.data:  # Check if xyxy is the correct attribute for bounding box coordinates
            print(box)
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(x1, y1, f'{result.names[int(cls)]}: {conf:.2f}', color='white', fontsize=12,
                     bbox=dict(facecolor='red', alpha=0.5))
        plt.savefig(os.path.join(save_dir, f'visualized{i}.png'))
        plt.show()
        plt.close()