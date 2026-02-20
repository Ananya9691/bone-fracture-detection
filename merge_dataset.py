import os
import shutil

source_dir = "archive"
target_dir = "data/train"

class_map = {
    "Mild": ["hairline", "greenstick", "avulsion"],
    "Moderate": ["oblique", "spiral", "longitudinal"],
    "Severe": ["comminuted", "compression", "dislocation"],
    "Other": ["impacted", "pathological", "intra"]
}

print("📂 Scanning dataset folders...\n")

all_folders = []
for root, dirs, files in os.walk(source_dir):
    for d in dirs:
        folder_path = os.path.join(root, d)
        all_folders.append(folder_path)

print("✅ Total folders found:", len(all_folders), "\n")

for new_class, keywords in class_map.items():
    os.makedirs(os.path.join(target_dir, new_class), exist_ok=True)

    for folder in all_folders:
        folder_name = os.path.basename(folder).lower()

        if any(keyword in folder_name for keyword in keywords):
            print(f"➡ Processing folder: {folder}")

            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)

                # ✅ Copy ONLY image files
                if os.path.isfile(item_path) and item.lower().endswith(('.png', '.jpg', '.jpeg')):
                    shutil.copy(
                        item_path,
                        os.path.join(target_dir, new_class, item)
                    )

print("\n✅ Dataset merged successfully!")
