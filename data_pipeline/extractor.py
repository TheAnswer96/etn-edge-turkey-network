import os
import cv2
import shutil
import random

VIDEO_DIR = "data/raw/video"
FRAME_DIR = "data/raw/frame"
SAMPLE_DIR = "data/raw/sample"

os.makedirs(FRAME_DIR, exist_ok=True)

def extract_frames_from_video(video_path, base_output_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(base_output_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)

        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_name} into {output_dir}")

def sample_random_frames(n_samples=1000):
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    # Get all image files from subdirectories
    all_images = []
    for root, dirs, files in os.walk(FRAME_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                all_images.append(os.path.join(root, f))

    total_images = len(all_images)

    if total_images == 0:
        print("No images found in frame directory or subdirectories.")
        return

    if total_images < n_samples:
        print(f"Only {total_images} images available. Copying all of them.")
        selected_images = all_images
    else:
        selected_images = random.sample(all_images, n_samples)

    for image_path in selected_images:
        image_name = os.path.basename(image_path)
        dst_path = os.path.join(SAMPLE_DIR, image_name)
        shutil.copy2(image_path, dst_path)

    print(f"Copied {len(selected_images)} images to {SAMPLE_DIR}")

def frame():
    for file in os.listdir(VIDEO_DIR):
        if file.lower().endswith(".mp4"):
            video_path = os.path.join(VIDEO_DIR, file)
            extract_frames_from_video(video_path, FRAME_DIR)

def sample():
    sample_random_frames(1000)


def retrieve_annotated_images(angle):
    angle = str(angle)

    labels_dir = os.path.join("src", "raw", "preimages", angle, "labels")
    frames_dir = os.path.join("src", "raw", "frame", angle)
    output_images_dir = os.path.join("src", "raw", "preimages", angle, "images")

    os.makedirs(output_images_dir, exist_ok=True)

    if not os.path.exists(labels_dir):
        print(f"Labels folder not found: {labels_dir}")
        return

    if not os.path.exists(frames_dir):
        print(f"Frames folder not found: {frames_dir}")
        return

    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]

    if not label_files:
        print("No labels files found.")
        return

    copied = 0
    missing = 0

    for label_file in label_files:
        base_name = os.path.splitext(label_file)[0]

        # Try common image extensions
        possible_extensions = [".jpg", ".jpeg", ".png"]
        image_found = False

        for ext in possible_extensions:
            image_name = base_name + ext
            src_image_path = os.path.join(frames_dir, image_name)

            if os.path.exists(src_image_path):
                dst_image_path = os.path.join(output_images_dir, image_name)
                shutil.copy2(src_image_path, dst_image_path)
                copied += 1
                image_found = True
                break

        if not image_found:
            missing += 1

    print(f"Angle: {angle}")
    print(f"Annotated images copied: {copied}")
    if missing > 0:
        print(f"Warning: {missing} labels had no matching image")



def retrieve_annotated_xml():
    labels_dir = os.path.join("src", "dataset", "xml_labels")
    frames_dir = os.path.join("src", "raw", "frame", "90")
    output_images_dir = os.path.join("src", "dataset", "images")

    os.makedirs(output_images_dir, exist_ok=True)

    if not os.path.exists(labels_dir):
        print(f"Labels folder not found: {labels_dir}")
        return

    if not os.path.exists(frames_dir):
        print(f"Frames folder not found: {frames_dir}")
        return

    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".xml")]

    if not label_files:
        print("No labels files found.")
        return

    copied = 0
    missing = 0

    for label_file in label_files:
        base_name = os.path.splitext(label_file)[0]

        # Try common image extensions
        possible_extensions = [".jpg", ".jpeg", ".png"]
        image_found = False

        for ext in possible_extensions:
            image_name = base_name + ext
            src_image_path = os.path.join(frames_dir, image_name)

            if os.path.exists(src_image_path):
                dst_image_path = os.path.join(output_images_dir, image_name)
                shutil.copy2(src_image_path, dst_image_path)
                copied += 1
                image_found = True
                break

        if not image_found:
            missing += 1

    print(f"Annotated images copied: {copied}")
    if missing > 0:
        print(f"Warning: {missing} labels had no matching image")
