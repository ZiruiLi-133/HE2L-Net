from ultralytics import YOLO
import torch
import os
from configs import cfgs
from matplotlib import pyplot as plt
import cv2

if __name__ == '__main__':
    # print(torch.cuda.is_available())
    # print(torch.cuda.get_device_name(0))
    # model = YOLO('/home/zli133/ECE_208/HE2L-Net/outputs/yolo_runs/train6/weights/last.pt')
    model = YOLO('/home/zli133/ECE_208/HE2L-Net/outputs/yolo_runs/train6/weights/last.pt')

    save_dir = os.path.join(cfgs.OUTPUTS.root, 'yolo_runs')
    # results = model.train(data=cfgs.TRAIN_Rec.DATASET.config, epochs=cfgs.TRAIN_Rec.epochs, imgsz=cfgs.TRAIN_Rec.image_size, project=save_dir, device=0)

    image_names = range(10)
    image_root = cfgs.TRAIN_Rec.DATASET.test_img
    image_path = [os.path.join(image_root, str(80000+image_name)+'.jpg') for image_name in image_names]
    pred_result = model.predict(image_path, show=True, save=True)
    # pred_result = model(image_path)
    ## Visualization
    # for i, result in enumerate(pred_result):
    #     rgb_img = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
    #     plt.imshow(rgb_img)
    #     plt.axis('off')
    #     print(result.boxes)

    #     for box in result.boxes.data:  # Check if xyxy is the correct attribute for bounding box coordinates
    #         print(box)
    #         x1, y1, x2, y2, conf, cls = box.cpu().numpy()
    #         rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=1)
    #         plt.gca().add_patch(rect)
    #         plt.text(x1, y1, f'{result.names[int(cls)]}: {conf:.2f}', color='white', fontsize=6,
    #                  bbox=dict(facecolor='red', alpha=0.5))
    #     plt.savefig(os.path.join(save_dir, f'test_visualized{i}.png'))
    #     plt.close()