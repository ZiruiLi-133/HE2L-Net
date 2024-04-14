import cv2
import matplotlib.pyplot as plt
import os

import cv2
import matplotlib.pyplot as plt
import os

def draw_bbox(img, bbox, label, color):
    """ Draw a bounding box with a label on the image. """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def visualize_dataset(path, img_folder, text_folder):
    class_labels = [
        "lim_", "a", "to", "frac", "pi", "4", "d", "left(", "sin", "+", "-", "6", "sec", "right)",
        "w", "/", "5", "tan", "2", "3", "e", "b", "7", "cos", "theta", "8", "=", "x", "9", "1",
        "y", "h", "k", "g", "csc", "infty", "0", "sqrt", "r", "ln", "n", "u", "cot", "left|", "right|",
        "p", "t", "z", "log", "v", "s", "c", "cdot", "."
    ]
    images_dir = os.path.join(path, img_folder)
    text_dir = os.path.join(path, text_folder)
    files = os.listdir(images_dir)
    print(files)
    i = 0
    for file in files:
        if i == 5:
            break
        if file.endswith(".jpg"):
            # Read image
            img_path = os.path.join(images_dir, file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Corresponding annotation file
            txt_file = file.replace(".jpg", ".txt")
            txt_path = os.path.join(text_dir, txt_file)
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])

                        # Convert from normalized to absolute coordinates
                        abs_width, abs_height = img.shape[1], img.shape[0]
                        x1 = int((x_center - width / 2) * abs_width)
                        y1 = int((y_center - height / 2) * abs_height)
                        x2 = int((x_center + width / 2) * abs_width)
                        y2 = int((y_center + height / 2) * abs_height)

                        draw_bbox(img, (x1, y1, x2, y2), class_labels[class_id], (255, 0, 0))

            # Plot the image
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.show()
            i += 1

# Example usage
path = 'D:\Machine_Learning_Projects\HE2L-Net\datasets\kaggle_data_coco'
img_folder = 'images'
text_folder = 'labels'
visualize_dataset(path, img_folder, text_folder)
