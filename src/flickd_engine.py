from typing import Dict, List, Any
import json
from pathlib import Path
import uuid
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO



from .video_processor import VideoProcessor
from .object_detector import ObjectDetector
from .product_matcher import ProductMatcher
from .vibe_classifier import VibeClassifier

class FlickdEngine:
    def __init__(self):
        """Initialize all components of the Flickd engine."""
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

    def process_video(self, 
                     video_path: str, 
                     caption: str = "", 
                     hashtags: List[str] = None,
                     transcript: str = "") -> Dict[str, Any]:
        """
        Process a video and return structured data about detected products and vibes.
        
        Args:
            video_path: Path to video file
            caption: Video caption text
            hashtags: List of hashtags
            transcript: Video transcript text
        
        Returns:
            Dictionary containing video analysis results
        """
        # Generate video ID
        video_id = str(uuid.uuid4())
        
        # Process text for vibe classification
        text_content = " ".join(filter(None, [
            caption,
            " ".join(hashtags) if hashtags else "",
            transcript
        ]))
        vibes = self.vibe_classifier.classify_vibes(text_content)
        
        # Process video frames
        products = []
        with VideoProcessor(video_path) as video_proc:
            for frame_number, frame in video_proc.extract_frames():
                # Detect objects in frame
                detections = self.object_detector.detect_objects(frame)
                
                for detection in detections:
                    # Crop detected object
                    cropped_obj = self.object_detector.crop_detection(frame, detection["bbox"])
                    
                    # Match product
                    match_result = self.product_matcher.match_product(cropped_obj)
                    
                    if match_result["match_type"] != "no_match":
                        product_info = {
                            "frame_number": frame_number,
                            "timestamp": video_proc.get_frame_timestamp(frame_number),
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
            "id": video_id,
            "type": "video",
            "vibes": vibes,
            "products": products
        }
        
        return result

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save analysis results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2) 