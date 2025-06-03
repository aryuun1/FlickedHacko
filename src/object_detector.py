from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any
from .config import YOLO_MODEL_NAME, FASHION_CLASSES, MODELS_DIR

class ObjectDetector:
    def __init__(self):
        """Initialize YOLO model for fashion item detection."""
        print("Initializing YOLO model...")
        model_path = MODELS_DIR / YOLO_MODEL_NAME
        self.model = YOLO(model_path if model_path.exists() else YOLO_MODEL_NAME)
        print("YOLO model initialized")
        
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect fashion items in a frame.
        Returns list of detections with class, bbox, and confidence.
        """
        print("\nDetecting objects in frame...")
        results = self.model(frame, conf=0.15)[0]  # Lower confidence for initial detection
        detections = []

        print(f"Found {len(results.boxes)} potential objects")
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            confidence = float(box.conf[0])
            
            # Only process if class is in our fashion categories
            if class_name.lower() in [c.lower() for c in FASHION_CLASSES]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
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
                print(f"Detected {class_name} with confidence {confidence:.2f}")

        print(f"Found {len(detections)} fashion items")
        return detections

    def crop_detection(self, frame: np.ndarray, bbox: Dict[str, int]) -> np.ndarray:
        """Crop detected object from frame using bbox."""
        x, y = bbox["x"], bbox["y"]
        w, h = bbox["width"], bbox["height"]
        
        # Add padding to the crop
        pad = 10
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2*pad)
        h = min(frame.shape[0] - y, h + 2*pad)
        
        return frame[y:y+h, x:x+w] 