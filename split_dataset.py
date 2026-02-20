import os
import shutil
import random

base_dir = "data/train"
val_dir = "data/val"
test_dir = "data/test"

split_ratio = 0.15  # 15% validation, 15% test

for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    images = os.listdir(class_path)

    random.shuffle(images)

    val_count = int(len(images) * split_ratio)
    test_count = int(len(images) * split_ratio)

    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Move to validation
    for img in images[:val_count]:
        shutil.move(
            os.path.join(class_path, img),
            os.path.join(val_dir, class_name, img)
        )

    # Move to test
    for img in images[val_count:val_count + test_count]:
        shutil.move(
            os.path.join(class_path, img),
            os.path.join(test_dir, class_name, img)
        )

print("✅ Dataset split completed!")
