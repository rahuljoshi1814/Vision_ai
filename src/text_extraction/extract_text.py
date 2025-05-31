import easyocr
import cv2
import os
from typing import Dict

def extract_text_from_images(folder_path: str) -> Dict[str, str]:
    """
    Extracts text from all images in a folder using EasyOCR.

    Args:
        folder_path (str): Path to folder with segmented images.

    Returns:
        Dict[str, str]: Mapping from image filename to extracted text.
    """
    reader = easyocr.Reader(['en'])  # Load EasyOCR reader for English
    results = {}

    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder_path, file)
            image = cv2.imread(path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Use EasyOCR on the numpy image (gray)
            result = reader.readtext(gray)
            extracted_text = " ".join([res[1] for res in result])
            results[file] = extracted_text.strip()

    return results
