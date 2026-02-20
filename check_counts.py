import os

base_path = "data/train"

for cls in os.listdir(base_path):
    cls_path = os.path.join(base_path, cls)
    if os.path.isdir(cls_path):
        print(cls, "→", len(os.listdir(cls_path)), "images")
