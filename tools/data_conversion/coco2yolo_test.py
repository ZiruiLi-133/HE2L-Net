import json
import os
from pathlib import Path
from configs import cfgs
import requests
import yaml
from PIL import Image
from tqdm import tqdm
import shutil

def make_dirs(dir="new_dir/"):
    """Creates a directory with subdirectories 'labels' and 'images', removing existing ones."""
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)  # delete dir
    for p in dir, dir / "labels", dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir


def convert(file, zip=False):
    """Converts Labelbox JSON labels to YOLO format and saves them, with optional zipping."""
    names = []  # class names
    names_dict_path = '/home/zli133/ECE_208/HE2L-Net/datasets/calculus_dataset_com/symbol_to_id.json'
    with open(names_dict_path) as f:
        names_dict = json.load(f)
    for k, v in names_dict.items():
        names.append(k)
    file = Path(file)
    print(names)
    # Assuming make_dirs creates a directory from file.stem and returns its Path object
    save_dir = make_dirs(file.parent / file.stem)  # Use parent directory and file.stem for new directory

    with open(file) as f:
        data = json.load(f)  # load JSON
    for img in tqdm(data['images'], desc=f"Converting {file}"):
        # im_path = os.path.join(cfgs.DATASETS.root, 'calculus_images', img["file_name"])
        width = img['width']
        height = img['height']
        # im = Image.open(requests.get(im_path, stream=True).raw if im_path.startswith("http") else im_path)  # open
        # width, height = im.size  # image size
        label_path = save_dir / "labels" / Path(str(img["id"])).with_suffix(".txt").name
        image_path = save_dir / "images" / Path(str(img["id"])).with_suffix(".jpg")
        # im.save(image_path, quality=95, subsampling=0)

        labels = [a for a in data['annotations'] if a['image_id'] == img["id"]]
        for label in labels:
            # box
            left, top, w, h = label["bbox"]  # top, left, height, width
            xywh = [(left + w / 2) / width, (top + h / 2) / height, w / width, h / height]  # xywh normalized

            # class
            cls_num = label["category_id"]  # class name
            cls = data['categories'][cls_num]['name']
            # print(f'{cls} : {names.index(cls)}')
            line = names.index(cls), *xywh  # YOLO format (class_index, xywh)
            with open(label_path, "a") as f:
                f.write(("%g " * len(line)).rstrip() % line + "\n")

    # Save dataset.yaml
    d = {
        "path": f"../datasets/{file.stem}  # dataset root dir",
        "train": "images/train  # train images (relative to path) 128 images",
        "val": "images/val  # val images (relative to path) 128 images",
        "test": " # test images (optional)",
        "nc": len(names),
        "names": names,
    }  # dictionary

    with open(save_dir / file.with_suffix(".yaml").name, "w") as f:
        yaml.dump(d, f, sort_keys=False)

    # Zip
    if zip:
        print(f"Zipping as {save_dir}.zip...")
        os.system(f"zip -qr {save_dir}.zip {save_dir}")

    print("Conversion completed successfully!")


if __name__ == "__main__":
    convert("/gpfs/fs1/home/zli133/ECE_208/HE2L-Net/datasets/calculus_dataset_yolo/calculus_test.json")