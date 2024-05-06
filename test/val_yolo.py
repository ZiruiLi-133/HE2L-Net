from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('/home/zli133/ECE_208/HE2L-Net/outputs/yolo_runs/train11/weights/best.pt')
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    print(metrics.box.mp)
    print(metrics.box.mr)
    print(metrics.box.map)    # map50-95
    print(metrics.box.map50)  # map50
    print(metrics.box.map75)  # map75
    print(metrics.box.maps)   # a list contains map50-95 of each category