from typing import Dict, List, Any
import json
from pathlib import Path
import uuid
import cv2
import numpy as np
from PIL import Image




from .video_processor import VideoProcessor
from .object_detector import ObjectDetector
from .product_matcher import ProductMatcher
from .vibe_classifier import VibeClassifier

class FlickdEngine:
    def __init__(self):
        """initialization of all the components within the engine"""
        self.object_detector = ObjectDetector()
        self.product_matcher = ProductMatcher()
        self.vibe_classifier = VibeClassifier()

    def process_image(self, 
                     image_path: str, 
                     caption: str = "", 
                     hashtags: List[str] = None) -> Dict[str, Any]:
        """
        Process a single image and return structured data about detected products and vibes.
        
        Args:
            image_path: Path to image file
            caption: Image caption text
            hashtags: List of hashtags
        
        Returns:
            Dictionary containing image analysis results
        """
        # Generate ID
        image_id = str(uuid.uuid4())
        
        # Process text for vibe classification
        text_content = " ".join(filter(None, [
            caption,
            " ".join(hashtags) if hashtags else ""
        ]))
        vibes = self.vibe_classifier.classify_vibes(text_content)
        
        # Read and process image
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Detect objects in image
        detections = self.object_detector.detect_objects(frame)
        
        # Process detections
        products = []
        for detection in detections:
            # Crop detected object
            cropped_obj = self.object_detector.crop_detection(frame, detection["bbox"])
            
            # Match product
            match_result = self.product_matcher.match_product(cropped_obj)
            
            if match_result["match_type"] != "no_match":
                product_info = {
                    "type": detection["class_name"],
                    "bbox": detection["bbox"],
                    "detection_confidence": detection["confidence"],
                    "match_type": match_result["match_type"],
                    "matched_product_id": match_result["matched_product_id"],
                    "match_confidence": match_result["confidence"]
                }
                products.append(product_info)
        
        # Prepare final output
        result = {
            "id": image_id,
            "type": "image",
            "vibes": vibes,
            "products": products
        }
        
        return result

    def process_video(self, video_path: str, caption: str = "", hashtags: List[str] = None) -> Dict[str, any]:
        """Process video and return detected products and vibes."""
        print(f"\nProcessing: {video_path}")
        
        # Process text for vibe classification
        text_content = " ".join(filter(None, [
            caption,
            " ".join(hashtags) if hashtags else ""
        ]))
        
        # Process frames
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        # Track unique product matches to avoid duplicates
        unique_products = {}  # product_id -> {count, confidence, type}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % 30 == 0:  # Process every 30th frame
                # Detect objects
                detections = self.object_detector.detect_objects(frame)
                
                # Match each detection with products
                for detection in detections:
                    # Crop detected object
                    cropped_obj = self.object_detector.crop_detection(frame, detection["bbox"])
                    
                    # Match with product catalog
                    match_result = self.product_matcher.match_product(cropped_obj)
                    
                    if match_result["match_type"] != "no_match" and match_result["matched_product_id"]:
                        product_id = match_result["matched_product_id"]
                        if product_id not in unique_products:
                            unique_products[product_id] = {
                                "type": detection["class_name"],
                                "confidence": match_result["confidence"],
                                "count": 1
                            }
                        else:
                            # Update with higher confidence if found
                            if match_result["confidence"] > unique_products[product_id]["confidence"]:
                                unique_products[product_id]["confidence"] = match_result["confidence"]
                            unique_products[product_id]["count"] += 1
            
            frame_count += 1
            
        cap.release()
        
        # Get top products (those seen multiple times with high confidence)
        top_products = []
        for product_id, info in unique_products.items():
            if info["count"] >= 2 and info["confidence"] >= 0.6:  # Must be seen at least twice with good confidence
                top_products.append({
                    "product_id": product_id,
                    "type": info["type"],
                    "match_confidence": info["confidence"]
                })
        
        # Sort by confidence and take top 5
        top_products = sorted(top_products, key=lambda x: x["match_confidence"], reverse=True)[:5]
        
        # Get vibes
        vibes = self.vibe_classifier.classify_vibes(text_content)
        
        return {
            "id": str(uuid.uuid4()),
            "type": "video",
            "vibes": vibes,
            "products": top_products
        }

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save analysis results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2) 