import os
import json
import argparse
from src.segmentation.segment_image import segment_image
from src.object_analysis.detect_objects import classify_images
from src.text_extraction.extract_text import extract_text_from_images
from src.summarization.summarize_data import summarize_objects
import cv2 

def main(input_image_path: str):
    print("🔹 Starting AI Vision Pipeline...")

    # 1. Create necessary output dirs
    segmented_dir = "data/segmented"
    results_path = "data/results/final_summary.json"
    os.makedirs(segmented_dir, exist_ok=True)
    os.makedirs("data/results", exist_ok=True)

    # 2. Step 1: Segment image
    print(" Segmenting input image...")
    segmented_paths = segment_image(image_path=input_image_path, output_dir=segmented_dir)
    if not segmented_paths:
        print(" No objects found in the image.")
        return

    # 3. Step 2: Classify segmented objects
    print(" Classifying segmented objects...")
    classifications = classify_images(segmented_dir)

    # 4. Step 3: Extract text from each object
    print(" Extracting OCR text from segmented objects...")
    ocr_texts = extract_text_from_images(segmented_dir)

    # 5. Step 4: Summarize each object's results
    print(" Generating summaries...")
    summaries = summarize_objects(classifications, ocr_texts)

    # 6. Save output
    with open(results_path, "w") as f:
        json.dump(summaries, f, indent=4)
    print(f" Pipeline complete. Results saved to `{results_path}`")

    # Optional: Display summaries
    for img, summary in summaries.items():
        print(f"\n{img} ➜ {summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AI Vision Pipeline.")
    parser.add_argument("image", nargs="?", default="data/raw/image.jpeg", help="Path to input image")
    args = parser.parse_args()
    main(args.image)
