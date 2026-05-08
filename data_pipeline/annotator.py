import os
import shutil
import json
from ultralytics import YOLO
from PIL import Image
import xml.etree.ElementTree as ET
import random
from pathlib import Path

BASE_RAW_DIR = "data/raw"


def get_paths(angle):
    angle = str(angle)

    paths = {
        "sample_dir": os.path.join(BASE_RAW_DIR, "frame", angle),  # prediction source
        "labels_dir": os.path.join(BASE_RAW_DIR, "preimages", angle, "labels"),
        "preimages_dir": os.path.join(BASE_RAW_DIR, "preimages", angle, "images"),
        "dataset_yaml": os.path.join(BASE_RAW_DIR, f"dataset_{angle}.yaml"),
        "root_dataset_path": os.path.join(BASE_RAW_DIR, "preimages", angle),
    }

    return paths


# -----------------------
# Step 1: Copy annotated images (angle-aware)
# -----------------------

def copy_annotated_images(angle):
    paths = get_paths(angle)

    LABELS_DIR = paths["labels_dir"]
    SAMPLE_DIR = paths["sample_dir"]
    PREIMAGES_DIR = paths["preimages_dir"]

    os.makedirs(PREIMAGES_DIR, exist_ok=True)

    if not os.path.exists(LABELS_DIR):
        print(f"Labels folder not found: {LABELS_DIR}")
        return

    label_files = [
        f for f in os.listdir(LABELS_DIR)
        if f.endswith(".txt")
    ]

    copied = 0
    missing = 0

    for label_file in label_files:
        base_name = os.path.splitext(label_file)[0]

        # Support multiple image formats
        found = False
        for ext in [".jpg", ".jpeg", ".png"]:
            image_name = base_name + ext
            src_image_path = os.path.join(SAMPLE_DIR, image_name)

            if os.path.exists(src_image_path):
                dst_image_path = os.path.join(PREIMAGES_DIR, image_name)
                shutil.copy2(src_image_path, dst_image_path)
                copied += 1
                found = True
                break

        if not found:
            missing += 1

    print(f"[Angle {angle}] Copied {copied} annotated images to {PREIMAGES_DIR}")
    if missing > 0:
        print(f"[Angle {angle}] Missing images for {missing} labels")


# -----------------------
# Step 2: Create dataset.yaml (angle-aware)
# -----------------------

def create_dataset_yaml(angle):
    paths = get_paths(angle)

    dataset_yaml_path = paths["dataset_yaml"]
    root_dataset_path = paths["root_dataset_path"]

    content = f"""
path: {root_dataset_path}
train: images
val: images

names:
  0: body
  1: neck
"""

    with open(dataset_yaml_path, "w") as f:
        f.write(content.strip())

    print(f"Created dataset YAML: {dataset_yaml_path}")
    return dataset_yaml_path


# -----------------------
# Step 3: Train YOLOv11 (angle-aware)
# -----------------------

def train_yolo(angle, epochs=50, model_size="n", img_size=640):
    """
    angle: 45 or 90
    model_size options:
        n  -> nano
        s  -> small
        m  -> medium
        l  -> large
        x  -> extra large
    """

    dataset_yaml = create_dataset_yaml(angle)

    model_name = f"yolo11{model_size}.pt"
    model = YOLO(model_name)

    model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=8
    )

    print(f"Training completed for angle {angle}")
    return model


# -----------------------
# Step 4: Predict and Export COCO JSON (angle-aware)
# -----------------------

def predict_and_export(model, angle, output_json=None, conf_thres=0.25):
    paths = get_paths(angle)
    sample_dir = paths["sample_dir"]

    if output_json is None:
        output_json = os.path.join(
            "src", "raw", f"predictions_coco_{angle}.json"
        )

    if not os.path.exists(sample_dir):
        print(f"Sample directory not found: {sample_dir}")
        return

    image_files = sorted([
        f for f in os.listdir(sample_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if len(image_files) == 0:
        print("No images found for prediction.")
        return

    coco_output = {
        "info": {
            "description": f"YOLO predictions for angle {angle}",
            "version": "1.0"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "body", "supercategory": "person"},
            {"id": 1, "name": "neck", "supercategory": "person"}
        ]
    }

    annotation_id = 0

    for image_id, image_name in enumerate(image_files):
        image_path = os.path.join(sample_dir, image_name)

        # Read image size (required for COCO)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Skipping corrupted image: {image_name} ({e})")
            continue

        coco_output["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })

        results = model.predict(
            source=image_path,
            conf=conf_thres,
            save=False,
            verbose=False
        )

        if not results:
            continue

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            score = float(box.conf[0])
            if score < conf_thres:
                continue

            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area = w * h

            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cls_id,
                "bbox": [x1, y1, w, h],  # COCO format
                "area": area,
                "score": score,
                "iscrowd": 0,
                "segmentation": []
            })

            annotation_id += 1

        if image_id % 100 == 0:
            print(f"Processed {image_id + 1}/{len(image_files)} images...")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w") as f:
        json.dump(coco_output, f, indent=4)

    print(f"Saved COCO annotations: {output_json}")
    print(f"Total images: {len(coco_output['images'])}")
    print(f"Total detections: {len(coco_output['annotations'])}")


# -----------------------
# Main pipeline (angle parametric)
# -----------------------

def helper(angle=90, epochs=150, model_size="n", img_size=640):

    # Step 1: gather annotated images for this angle
    # copy_annotated_images(angle)

    # Step 2: train
    # model = train_yolo(
    #     angle=angle,
    #     epochs=epochs,
    #     model_size=model_size,
    #     img_size=img_size,
    # )

    # Optional: reload best weights (as in your original code)
    best_model_path = "runs/detect/train3/weights/best.pt"
    if os.path.exists(best_model_path):
        model = YOLO(best_model_path)

    # Step 3: predict on full frame set for that angle
    predict_and_export(model, angle)



def convert_voc_to_yolo(xml_dir="src/dataset/xml_labels",
                            output_dir="src/dataset/labels",
                            class_map=None):
    if class_map is None:
        class_map = {"body": 0, "neck": 1}

    os.makedirs(output_dir, exist_ok=True)

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        img_width = float(size.find("width").text)
        img_height = float(size.find("height").text)

        yolo_lines = []

        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip()
            if class_name not in class_map:
                continue

            class_id = class_map[class_name]

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            x_center = ((xmin + xmax) / 2.0) / img_width
            y_center = ((ymin + ymax) / 2.0) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        txt_filename = os.path.splitext(xml_file)[0] + ".txt"
        output_path = os.path.join(output_dir, txt_filename)

        with open(output_path, "w") as f:
            f.write("\n".join(yolo_lines))

def split_dataset_yolo(
    images_dir="src/dataset/images",
    yolo_labels_dir="src/dataset/labels",
    xml_labels_dir="src/dataset/xml_labels",
    output_root="src/dataset_split",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    random.seed(seed)

    images_dir = Path(images_dir)
    yolo_labels_dir = Path(yolo_labels_dir)
    xml_labels_dir = Path(xml_labels_dir)
    output_root = Path(output_root)

    splits = ["train", "val", "test"]

    for split in splits:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)
        (output_root / split / "xml").mkdir(parents=True, exist_ok=True)

    # Collect images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = [
        f for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    random.shuffle(image_files)
    total = len(image_files)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    split_mapping = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:]
    }

    for split_name, files in split_mapping.items():
        for img_path in files:
            stem = img_path.stem

            yolo_label_path = yolo_labels_dir / f"{stem}.txt"
            xml_label_path = xml_labels_dir / f"{stem}.xml"

            dest_img = output_root / split_name / "images" / img_path.name
            dest_yolo = output_root / split_name / "labels" / f"{stem}.txt"
            dest_xml = output_root / split_name / "xml" / f"{stem}.xml"

            shutil.copy2(img_path, dest_img)

            if yolo_label_path.exists():
                shutil.copy2(yolo_label_path, dest_yolo)

            if xml_label_path.exists():
                shutil.copy2(xml_label_path, dest_xml)

    print("Dataset successfully split into train/val/test.")
    print(f"Total images: {total}")
    print(f"Train: {len(split_mapping['train'])}")
    print(f"Val: {len(split_mapping['val'])}")
    print(f"Test: {len(split_mapping['test'])}")
