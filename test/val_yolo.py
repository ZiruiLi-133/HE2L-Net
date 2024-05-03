from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('/home/zli133/ECE_208/HE2L-Net/outputs/yolo_runs/train10/weights/best.pt')
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category