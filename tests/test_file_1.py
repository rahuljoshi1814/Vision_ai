# test_object_analysis.py
from src.object_analysis.detect_objects import classify_images

folder_path = "data/segmented"
results = classify_images(folder_path)

for img, label in results.items():
    print(f"{img} ➜ {label}")
