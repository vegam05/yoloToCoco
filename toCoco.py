#Script to convert yolo annoatations and images to coco format
import os
import json
from PIL import Image, ExifTags#Some jpeg images have exif metadata, which results in swapped dimensions in coco annotation json, Hence handling this is necessary

def get_image_size_with_orientation(img_path):
    with Image.open(img_path) as img:
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(img._getexif().items())
            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img = img.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            pass
        return img.size

def convert_yolo_to_coco(dataset_dir, class_names):
    coco_train = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    coco_val = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    for i, class_name in enumerate(class_names):
        category_entry = {
            "id": i + 1,
            "name": class_name,
            "supercategory": "none"
        }
        coco_train["categories"].append(category_entry)
        coco_val["categories"].append(category_entry)

    annotation_id_train = 1
    annotation_id_val = 1

    for subset in ["train", "val"]:
        subset_image_dir = os.path.join(dataset_dir, 'images', subset)
        subset_label_dir = os.path.join(dataset_dir, 'labels', subset)
        coco_subset = coco_train if subset == "train" else coco_val
        annotation_id = annotation_id_train if subset == "train" else annotation_id_val

        for img_file in os.listdir(subset_image_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_id = os.path.splitext(img_file)[0]
            img_path = os.path.join(subset_image_dir, img_file)
            label_path = os.path.join(subset_label_dir, img_id + '.txt')

            width, height = get_image_size_with_orientation(img_path)

            coco_subset["images"].append({
                "id": img_id,
                "width": width,
                "height": height,
                "file_name": os.path.join(subset, img_file)
            })

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        if line.strip() == '':
                            continue
                        class_id, x_center, y_center, w, h = map(float, line.strip().split())
                        
                        x_center *= width
                        y_center *= height
                        w *= width
                        h *= height
                        
                        x_min = x_center - w / 2
                        y_min = y_center - h / 2

                        coco_subset["annotations"].append({
                            "id": annotation_id,
                            "image_id": img_id,
                            "category_id": int(class_id) + 1,
                            "bbox": [x_min, y_min, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        annotation_id += 1

            if subset == "train":
                annotation_id_train = annotation_id
            else:
                annotation_id_val = annotation_id

    with open(os.path.join(dataset_dir, 'annotations', 'instances_train.json'), 'w') as f:
        json.dump(coco_train, f, indent=4)

    with open(os.path.join(dataset_dir, 'annotations', 'instances_val.json'), 'w') as f:
        json.dump(coco_val, f, indent=4)

class_names = ['car', 'bike', 'auto', 'rickshaw', 'cycle', 'bus', 'minitruck', 'truck', 'van', 'taxi', 'motorvan', 'toto', 'train', 'boat', 'cycle van']  # Replace with your actual class names

dataset_dir = "dataset" #The dataset directory should have images and labels directories with train and val subdirectories

convert_yolo_to_coco(dataset_dir, class_names)

