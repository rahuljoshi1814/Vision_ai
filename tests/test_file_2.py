# test_text_extraction.py
from src.text_extraction.extract_text import extract_text_from_images

folder = "data/segmented"
ocr_results = extract_text_from_images(folder)

for img, text in ocr_results.items():
    print(f"{img} ➜ {text}")
