# test_summarization.py
from src.object_analysis.detect_objects import classify_images
from src.text_extraction.extract_text import extract_text_from_images
from src.summarization.summarize_data import summarize_objects

folder = "data/segmented"

classifications = classify_images(folder)
ocr_texts = extract_text_from_images(folder)
summaries = summarize_objects(classifications, ocr_texts)

for img, summary in summaries.items():
    print(f"{img} ➜ {summary}\n")
