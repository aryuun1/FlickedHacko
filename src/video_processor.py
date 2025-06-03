import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple
from .config import FRAME_EXTRACTION_INTERVAL

class VideoProcessor:
    def __init__(self, video_path: str):
        """Initialize video processor with path to video file."""
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps

    def extract_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Extract frames from video at specified interval.
        Yields: (frame_number, frame_data)
        """
        frame_number = 0
        frames_to_skip = int(self.fps * FRAME_EXTRACTION_INTERVAL)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_number % frames_to_skip == 0:
                yield frame_number, frame

            frame_number += 1

    def get_frame_timestamp(self, frame_number: int) -> float:
        """Convert frame number to timestamp in seconds."""
        return frame_number / self.fps

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release() 