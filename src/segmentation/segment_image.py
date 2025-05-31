import cv2
import numpy as np
import os
from typing import List, Tuple

def segment_image(image_path: str, output_dir: str = None, debug: bool = False) -> List[str]:
    """
    Segments objects in an image using basic contour detection.
    
    Args:
        image_path (str): Path to the input image.
        output_dir (str): If provided, saves segmented objects to this directory.
        debug (bool): If True, shows intermediate results.
        
    Returns:
        List[str]: List of file paths to segmented object images.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 150)

    if debug:
        cv2.imshow("Edged", edged)
        cv2.waitKey(0)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    segmented_paths = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 30 and h > 30:  # Filter out small noise
            roi = image[y:y+h, x:x+w]
            output_path = os.path.join(output_dir, f"object_{i+1}.png") if output_dir else None
            if output_path:
                cv2.imwrite(output_path, roi)
                segmented_paths.append(output_path)
    
    if debug:
        print(f"[INFO] Found {len(segmented_paths)} objects.")
    
    return segmented_paths
