from typing import Dict, List

def summarize_objects(
    classifications: Dict[str, str], 
    ocr_texts: Dict[str, str]
) -> Dict[str, str]:
    """
    Generate simple summaries combining classification and OCR results.

    Args:
        classifications (Dict[str, str]): Mapping of image to predicted class label.
        ocr_texts (Dict[str, str]): Mapping of image to extracted OCR text.

    Returns:
        Dict[str, str]: Mapping of image to summary text.
    """
    summaries = {}

    for img_name in classifications.keys():
        label = classifications.get(img_name, "Unknown object")
        ocr = ocr_texts.get(img_name, "")

        summary = f"This object is likely a **{label}**."
        if ocr:
            summary += f" It contains the text: \"{ocr}\"."
        else:
            summary += " No readable text found on the object."

        summaries[img_name] = summary

    return summaries
