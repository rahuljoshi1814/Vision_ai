from src.video_mode.video_pipeline import process_video

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

video_path = "data/raw/sample_video.mp4"
process_video(video_path)
