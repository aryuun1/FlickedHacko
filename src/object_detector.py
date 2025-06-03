from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any
from .config import YOLO_MODEL_NAME, FASHION_CLASSES, MODELS_DIR

class ObjectDetector:
    def __init__(self):
        """Initialize YOLO model for fashion item detection."""
        model_path = MODELS_DIR / YOLO_MODEL_NAME
        self.model = YOLO(model_path if model_path.exists() else YOLO_MODEL_NAME)
        
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect fashion items in a frame.
        Returns list of detections with class, bbox, and confidence.
        """
        results = self.model(frame, conf=0.25)[0]  # Lower confidence for initial detection
        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            
            # Only process if class is in our fashion categories
            if class_name.lower() in [c.lower() for c in FASHION_CLASSES]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                detection = {
                    "class_name": class_name,
                    "bbox": {
                        "x": int(x1),
                        "y": int(y1),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1)
                    },
                    "confidence": confidence
                }
                detections.append(detection)

        return detections

    def crop_detection(self, frame: np.ndarray, bbox: Dict[str, int]) -> np.ndarray:
        """Crop detected object from frame using bbox."""
        x, y = bbox["x"], bbox["y"]
        w, h = bbox["width"], bbox["height"]
        return frame[y:y+h, x:x+w] 