import cv2
import mediapipe as mp
import json
import shutil
from pathlib import Path
from tqdm import tqdm

mp_pose = mp.solutions.pose


def extract_keypoints_from_image(image_path, pose):
    """
    Extract keypoints from a single image using MediaPipe.
    Returns list of [x, y, visibility] or None if no detection.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return None

    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.append([lm.x, lm.y, lm.visibility])

    return keypoints


def process_dataset(input_dir, output_dir, failed_dir):
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    failed_root = Path(failed_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    failed_root.mkdir(parents=True, exist_ok=True)

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False
    )

    total = 0
    success = 0
    failed = 0
    failed_images = []

    class_dirs = [d for d in input_root.iterdir() if d.is_dir()]

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        output_class_dir = output_root / class_name
        failed_class_dir = failed_root / class_name

        output_class_dir.mkdir(parents=True, exist_ok=True)
        failed_class_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(class_dir.glob("*"))

        print(f"\nProcessing {class_name} ({len(image_files)} files)")

        for img_path in tqdm(image_files):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue

            total += 1
            keypoints = extract_keypoints_from_image(img_path, pose)

            if keypoints is None:
                failed += 1

                failed_images.append({
                    "class": class_name,
                    "image_name": img_path.name,
                    "image_path": str(img_path)
                })

                # copy failed image into failed folder
                shutil.copy2(img_path, failed_class_dir / img_path.name)
                continue

            data = {
                "label": class_name,
                "image": str(img_path),
                "image_name": img_path.name,
                "keypoints": keypoints
            }

            output_file = output_class_dir / (img_path.stem + ".json")
            with open(output_file, "w") as f:
                json.dump(data, f)

            success += 1

    # Save failure report
    failed_report_path = failed_root / "failed_images.json"
    with open(failed_report_path, "w") as f:
        json.dump(failed_images, f, indent=4)

    print("\n" + "=" * 50)
    print(f"Total images: {total}")
    print(f"Successful: {success}")
    print(f"Failed: {failed}")
    if total > 0:
        print(f"Success rate: {success / total * 100:.2f}%")

    print(f"\nFailed images saved to: {failed_report_path}")
    print(f"Failed image copies saved under: {failed_root}")


if __name__ == "__main__":
    input_dir = "data/raw/YogaPoses"
    output_dir = "keypoints"
    failed_dir = "failed_images"

    process_dataset(input_dir, output_dir, failed_dir)