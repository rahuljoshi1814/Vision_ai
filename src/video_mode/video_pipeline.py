import cv2
import os
import easyocr
from datetime import datetime
from src.object_analysis.yolo_detector import detect_objects_yolo

# Initialize OCR reader once
reader = easyocr.Reader(['en'])

def process_video(input_path: str, output_path: str = "data/results/output_video.avi", log_path: str = None):
    print(f"🎥 Processing video: {input_path}")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ Could not open video.")
        return

    # Get video details
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    frame_count = 0
    log_lines = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects_yolo(frame)
        frame_summary = [f"🖼️ Frame {frame_count + 1}:"]
        
        for label, (x1, y1, x2, y2) in detections:
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Crop and OCR
            roi = frame[y1:y2, x1:x2]
            try:
                ocr_result = reader.readtext(roi, detail=0)
                ocr_text = " ".join(ocr_result).strip()
            except:
                ocr_text = ""

            # Annotate
            display_text = f"{label} | {ocr_text}" if ocr_text else label
            cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Add to log summary
            frame_summary.append(f"  📦 {label} → \"{ocr_text}\"" if ocr_text else f"  📦 {label}")

        out.write(frame)
        log_lines.append("\n".join(frame_summary))
        frame_count += 1

    cap.release()
    out.release()
    print(f"✅ Processed {frame_count} frames. Output saved to: {output_path}")

    # Save log
    if log_path:
        with open(log_path, "w") as log_file:
            log_file.write(f"🕒 Timestamp: {datetime.now()}\n")
            log_file.write(f"🎥 Input Video: {os.path.basename(input_path)}\n")
            log_file.write(f"🎬 Output File: {os.path.basename(output_path)}\n")
            log_file.write(f"📸 Total Frames: {frame_count}\n\n")
            log_file.write("\n".join(log_lines))
