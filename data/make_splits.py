"""Create fixed train/val/test splits for yoga pose keypoint JSON files."""

import os
import json
import random


def make_splits(
    data_dir="../keypoints",
    output_dir="data/splits",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    # quick check
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must add up to 1.0")

    os.makedirs(output_dir, exist_ok=True)

    train_files = []
    val_files = []
    test_files = []

    # go through each class folder separately
    class_folders = sorted(
        [
            folder for folder in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, folder))
        ]
    )

    if not class_folders:
        raise ValueError(f"No class folders found in {data_dir}")

    print(f"Found {len(class_folders)} classes")

    for i, class_name in enumerate(class_folders):
        class_path = os.path.join(data_dir, class_name)

        # get all json files for this class
        files = sorted(
            [
                os.path.join(class_path, f)
                for f in os.listdir(class_path)
                if f.endswith(".json")
            ]
        )

        if not files:
            print(f"Skipping {class_name} (no json files found)")
            continue

        # use a fixed seed + class index so the split is reproducible
        rng = random.Random(seed + i)
        rng.shuffle(files)

        n = len(files)
        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)

        class_train = files[:train_end]
        class_val = files[train_end:val_end]
        class_test = files[val_end:]

        train_files.extend(class_train)
        val_files.extend(class_val)
        test_files.extend(class_test)

        print(
            f"{class_name}: total={n}, "
            f"train={len(class_train)}, val={len(class_val)}, test={len(class_test)}"
        )

    # shuffle final lists so batches are mixed across classes
    random.Random(seed).shuffle(train_files)
    random.Random(seed).shuffle(val_files)
    random.Random(seed).shuffle(test_files)

    # save to json
    with open(os.path.join(output_dir, "train_files.json"), "w") as f:
        json.dump(train_files, f, indent=2)

    with open(os.path.join(output_dir, "val_files.json"), "w") as f:
        json.dump(val_files, f, indent=2)

    with open(os.path.join(output_dir, "test_files.json"), "w") as f:
        json.dump(test_files, f, indent=2)

    print("\nDone.")
    print(f"Train samples: {len(train_files)}")
    print(f"Val samples:   {len(val_files)}")
    print(f"Test samples:  {len(test_files)}")
    print(f"Saved split files to: {output_dir}")


if __name__ == "__main__":
    make_splits()