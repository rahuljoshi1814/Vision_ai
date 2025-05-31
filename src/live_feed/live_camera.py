import sys
import os
import cv2

# ✅ Set root path BEFORE importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import cv2
import easyocr
from src.object_analysis.yolo_detector import detect_objects_yolo

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

def start_live_camera():
    print("🎥 Starting live camera... Press 'q' to exit.")
    
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("❌ Cannot access camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects_yolo(frame)

        for label, (x1, y1, x2, y2) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            roi = frame[y1:y2, x1:x2]
            try:
                ocr_result = reader.readtext(roi, detail=0)
                ocr_text = " ".join(ocr_result).strip()
            except:
                ocr_text = ""

            display_text = f"{label} | {ocr_text}" if ocr_text else label
            cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.namedWindow("Live Camera Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Live Camera Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Camera session ended.")

if __name__ == "__main__":
    start_live_camera()
