from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any
from .config import YOLO_MODEL_NAME, FASHION_CLASSES

# List of fashion-related classes that YOLO can detect
FASHION_CLASSES = [
    "person",  # For full-body shots
    "tie",
    "backpack",
    "umbrella",  # Could be a fashion accessory
    "handbag",
    "suitcase",
    "dress",
    "hat",
    "shoes",
    "eyeglasses",
    "shirt",
    "pants",
    "jacket",
    "coat",
    "skirt",
    "dress",
    "shorts",
    "suit",
    "scarf",
    "gloves",
    "boots",
    "sandals",
    "sneakers",
    "watch",
    "necklace",
    "bracelet",
    "earrings",
    "sunglasses",
    "belt",
    "purse",
    "wallet",
    "bag"
]

class ObjectDetector:
    def __init__(self):
        """Initialize YOLO model for fashion item detection."""
        print("Initializing YOLO model...")
       
        self.model = YOLO(YOLO_MODEL_NAME)
        print("YOLO model initialized")
        
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect fashion items in a frame.
        Returns list of detections with class, bbox, and confidence.
        """
        print("\nDetecting objects in frame...")
        results = self.model(frame, conf=0.01)[0]  # Even lower threshold
        detections = []

        print(f"Found {len(results.boxes)} potential objects")
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            confidence = float(box.conf[0])
            
            # Handle person detections specially - extract potential fashion items
            if class_name.lower() == "person":
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Create regions for potential fashion items
                # Upper body (shirt/top)
                upper_height = (y2 - y1) * 0.3
                upper_detection = {
                    "class_name": "shirt",
                    "bbox": {
                        "x": int(x1),
                        "y": int(y1),
                        "width": int(x2 - x1),
                        "height": int(upper_height)
                    },
                    "confidence": confidence * 0.8  # Slightly lower confidence
                }
                detections.append(upper_detection)
                print(f"Detected shirt from person with confidence {confidence * 0.8:.2f}")
                
                # Lower body (pants/skirt)
                lower_y = y1 + (y2 - y1) * 0.4  # Start below upper body
                lower_height = (y2 - y1) * 0.4
                lower_detection = {
                    "class_name": "pants",
                    "bbox": {
                        "x": int(x1),
                        "y": int(lower_y),
                        "width": int(x2 - x1),
                        "height": int(lower_height)
                    },
                    "confidence": confidence * 0.8
                }
                detections.append(lower_detection)
                print(f"Detected pants from person with confidence {confidence * 0.8:.2f}")
            
            # Process other fashion items normally
            elif class_name.lower() in [c.lower() for c in FASHION_CLASSES]:
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