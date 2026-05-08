import os
import csv
import glob
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
import pandas as pd


CLASSES = {
    "turkey": 0,
    "head": 1
}

XML_TO_CLASS = {
    "body": "turkey",
    "neck": "head"
}


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_image_path_map(images_dir):
    image_map = {}
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        for p in glob.glob(os.path.join(images_dir, ext)):
            name = os.path.splitext(os.path.basename(p))[0]
            image_map[name] = p
    return image_map


def parse_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Read image size safely
    size_node = root.find("size")
    img_w, img_h = None, None
    if size_node is not None:
        w_node = size_node.find("width")
        h_node = size_node.find("height")
        if w_node is not None and h_node is not None:
            img_w = float(w_node.text)
            img_h = float(h_node.text)

    boxes = []

    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is None:
            continue

        raw_name = name_node.text.strip().lower()

        # Map XML labels to our class system
        if raw_name not in XML_TO_CLASS:
            continue

        class_name = XML_TO_CLASS[raw_name]
        class_id = CLASSES[class_name]

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.findtext("xmin", default="0"))
        ymin = float(bndbox.findtext("ymin", default="0"))
        xmax = float(bndbox.findtext("xmax", default="0"))
        ymax = float(bndbox.findtext("ymax", default="0"))

        # Validate box
        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append({
            "class_name": class_name,
            "class_id": class_id,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "img_w": img_w,
            "img_h": img_h
        })

    return boxes, img_w, img_h


def bbox_to_xywh(box, img_w, img_h):
    x_center = ((box["xmin"] + box["xmax"]) / 2.0) / img_w
    y_center = ((box["ymin"] + box["ymax"]) / 2.0) / img_h
    width = (box["xmax"] - box["xmin"]) / img_w
    height = (box["ymax"] - box["ymin"]) / img_h
    return x_center, y_center, width, height


def perceptual_blur_metric(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # Apply 3x3 averaging filter (low-pass)
    blurred = cv2.blur(gray, (3, 3))

    # Compute absolute differences for original (horizontal and vertical)
    D_F_hor = np.abs(gray[:-1, :] - gray[1:, :])  # Vertical differences
    D_F_ver = np.abs(gray[:, :-1] - gray[:, 1:])  # Horizontal differences

    # Compute absolute differences for blurred
    D_B_hor = np.abs(blurred[:-1, :] - blurred[1:, :])
    D_B_ver = np.abs(blurred[:, :-1] - blurred[:, 1:])

    # Compute decreased variations (note: fixed to D_F - D_B)
    V_hor = np.maximum(D_F_hor - D_B_hor, 0)
    V_ver = np.maximum(D_F_ver - D_B_ver, 0)

    # Sum the differences
    s_F_hor = np.sum(D_F_hor)
    s_F_ver = np.sum(D_F_ver)
    s_V_hor = np.sum(V_hor)
    s_V_ver = np.sum(V_ver)

    # Avoid division by zero
    s_F_hor = max(s_F_hor, 1e-6)
    s_F_ver = max(s_F_ver, 1e-6)

    # Normalize to get blur scores (0: sharp, 1: blurry)
    b_hor = s_V_hor / s_F_hor  # Equivalent to 1 - (s_B_hor / s_F_hor), but corrected for metric direction
    b_ver = s_V_ver / s_F_ver

    # Final blur metric: max of horizontal and vertical
    blur = max(b_hor, b_ver)
    return blur


def compute_luminance_and_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    luminance = float(np.mean(gray))
    blur = perceptual_blur_metric(image)
    return luminance, blur


def get_instance_bucket(count):
    if count == 0:
        return "0"
    elif count <= 5:
        return "1-5"
    elif count <= 10:
        return "6-10"
    elif count <= 15:
        return "11-15"
    else:
        return ">15"


def compute_class_statistics(xml_dir):
    total_per_class = defaultdict(int)
    per_image_counts = []

    xml_files = os.listdir(xml_dir)

    for xml_file in xml_files:
        image_name = os.path.splitext(os.path.basename(xml_file))[0]
        boxes, _, _ = parse_voc_xml(os.path.join(xml_dir, xml_file))

        counts = {0: 0, 1: 0}
        for box in boxes:
            cid = box["class_id"]
            counts[cid] += 1
            total_per_class[cid] += 1

        per_image_counts.append({
            "image": image_name,
            "turkey_count": counts[0],
            "head_count": counts[1],
            "turkey_bucket": get_instance_bucket(counts[0]),
            "head_bucket": get_instance_bucket(counts[1])
        })

    return total_per_class, per_image_counts


def compute_iou(boxA, boxB):
    xA = max(boxA["xmin"], boxB["xmin"])
    yA = max(boxA["ymin"], boxB["ymin"])
    xB = min(boxA["xmax"], boxB["xmax"])
    yB = min(boxA["ymax"], boxB["ymax"])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA["xmax"] - boxA["xmin"]) * (boxA["ymax"] - boxA["ymin"])
    areaB = (boxB["xmax"] - boxB["xmin"]) * (boxB["ymax"] - boxB["ymin"])

    union = areaA + areaB - inter_area
    if union == 0:
        return 0.0

    return inter_area / union


def compute_per_image_average_iou(xml_dir):
    """
    Computes average IoU between ALL pairs of boxes of the same class per image,
    for each class in each image.
    """
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    per_image_avg_ious = []

    for xml_file in xml_files:
        image_name = os.path.splitext(os.path.basename(xml_file))[0]
        boxes, _, _ = parse_voc_xml(xml_file)

        # Group boxes by class
        class_groups = defaultdict(list)
        for b in boxes:
            class_groups[b["class_id"]].append(b)

        # Compute average pairwise IoU per class per image
        for class_id, bboxes in class_groups.items():
            n = len(bboxes)
            if n < 2:
                avg_iou = 0.0
            else:
                iou_sum = 0.0
                pair_count = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        iou = compute_iou(bboxes[i], bboxes[j])
                        iou_sum += iou
                        pair_count += 1
                avg_iou = iou_sum / pair_count

            class_name = [k for k, v in CLASSES.items() if v == class_id][0]
            per_image_avg_ious.append({
                "image": image_name,
                "class_name": class_name,
                "class_id": class_id,
                "avg_iou": avg_iou
            })

    return per_image_avg_ious


def export_bbox_csv(xml_dir, output_csv):
    rows = []
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))

    for xml_file in xml_files:
        image_name = os.path.splitext(os.path.basename(xml_file))[0]
        boxes, img_w, img_h = parse_voc_xml(xml_file)

        if img_w is None or img_h is None:
            continue

        for box in boxes:
            x_c, y_c, w, h = bbox_to_xywh(box, img_w, img_h)
            rows.append([image_name, box["class_name"], x_c, y_c, w, h])

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "image", "class", "x_center", "y_center", "width", "height"])
        for i, row in enumerate(rows):
            writer.writerow([i] + row)


def export_luminance_blur_csv(images_dir, xml_dir, output_csv):
    image_map = get_image_path_map(images_dir)
    rows = []
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))

    for xml_file in xml_files:
        image_name = os.path.splitext(os.path.basename(xml_file))[0]

        if image_name not in image_map:
            continue

        image = cv2.imread(image_map[image_name])
        if image is None:
            continue

        luminance, blur = compute_luminance_and_blur(image)
        n_lum = luminance / 255.0
        n_blur = blur  # Already normalized to [0,1]

        rows.append([image_name, n_lum, n_blur])

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "image", "normalized_luminance", "normalized_blurriness"])
        for i, row in enumerate(rows):
            writer.writerow([i] + row)


def export_class_counts_csv(total_per_class, per_image_counts, per_image_avg_ious, output_dir):
    summary_path = os.path.join(output_dir, "class_summary.csv")
    per_image_path = os.path.join(output_dir, "per_image_counts.csv")
    iou_path = os.path.join(output_dir, "per_image_average_iou.csv")

    # Summary counts
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "class", "class_id", "total_instances"])
        summary_rows = []
        for cname, cid in CLASSES.items():
            summary_rows.append([cname, cid, total_per_class[cid]])
        for i, row in enumerate(summary_rows):
            writer.writerow([i] + row)

    # Aggregated bucket instances
    bucket_sums = defaultdict(lambda: defaultdict(int))
    for row in per_image_counts:
        bucket_sums["turkey"][row["turkey_bucket"]] += row["turkey_count"]
        bucket_sums["head"][row["head_bucket"]] += row["head_count"]

    buckets = ["0", "1-5", "6-10", "11-15", ">15"]
    with open(per_image_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "bucket", "turkey_instances", "head_instances"
        ])
        for i, b in enumerate(buckets):
            writer.writerow([
                i, b,
                bucket_sums["turkey"][b],
                bucket_sums["head"][b]
            ])

    # Per-image average IoU per class
    with open(iou_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "image", "class", "class_id", "average_iou"])
        for i, row in enumerate(per_image_avg_ious):
            writer.writerow([
                i, row["image"], row["class_name"], row["class_id"], row["avg_iou"]
            ])


def run_full_analysis(
    images_dir=os.path.join(os.getcwd(), "src", "dataset", "images"),
    xml_dir=os.path.join(os.getcwd(), "src", "dataset", "xml_labels"),
    output_dir=os.path.join(os.getcwd(), "analysis")
):
    ensure_dir(output_dir)

    # 1) Instance statistics
    total_per_class, per_image_counts = compute_class_statistics(xml_dir)
    print("- class_summary.csv")
    # 2) Per-image average IoU per class
    per_image_avg_ious = compute_per_image_average_iou(xml_dir)
    print("- per_image_counts.csv")
    # 3) Export bbox CSV (per class)
    bbox_csv = os.path.join(output_dir, "bbox_xywh_per_class.csv")
    export_bbox_csv(xml_dir, bbox_csv)
    print("- bbox_xywh_per_class.csv")
    # 4) Export luminance & blur (per class)
    lum_blur_csv = os.path.join(output_dir, "luminance_blur_per_class.csv")
    export_luminance_blur_csv(images_dir, xml_dir, lum_blur_csv)
    print("- luminance_blur_per_class.csv")
    # 5) Export counts + IoU summaries
    export_class_counts_csv(total_per_class, per_image_counts, per_image_avg_ious, output_dir)
    print("- per_image_average_iou.csv")

def fix_csv():
    csv = pd.read_csv(os.path.join(os.getcwd(), "analysis", "bbox_xywh_per_class.csv"))
    mask1 = csv['class'] == 'turkey'
    turkey = csv[mask1]
    turkey = turkey.sample(n=2500)
    mask2 = csv['class'] == 'head'
    head = csv[mask2]
    head = head.sample(n=500)

    turkey.to_csv(os.path.join(os.getcwd(), "analysis", "bbox_xywh_per_turkey.csv"), index=False)
    head.to_csv(os.path.join(os.getcwd(), "analysis", "bbox_xywh_per_head.csv"), index=False)






