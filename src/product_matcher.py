from io import BytesIO
import torch
import faiss
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from .config import CLIP_MODEL_NAME, SIMILARITY_THRESHOLDS, CACHE_DIR
import cv2
import requests


class ProductMatcher:
    def __init__(self):
        """Initialize CLIP model and FAISS index for product matching."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.index = None
        self.product_ids = []

    def build_product_index(self, product_images: List[Dict[str, str]]):
        """
        Build FAISS index from product catalog images.
        product_images: List of dicts with 'id' and 'image_url' keys
        """
        embeddings = []
        self.product_ids = []

        for product in product_images:
            # Load and process image
            image = self._load_image_from_url(product['image_url'])
            if image is None:
                continue

            # Get image embedding
            embedding = self._get_image_embedding(image)
            embeddings.append(embedding)
            self.product_ids.append(product['id'])

        # Convert to numpy array and normalize
        embeddings = np.vstack(embeddings)
        faiss.normalize_L2(embeddings)

        # Build FAISS index
        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product = cosine similarity for normalized vectors
        self.index.add(embeddings)

    def match_product(self, image: np.ndarray) -> Dict[str, any]:
        """
        Match detected product against catalog.
        Returns match type and product ID.
        """
        if self.index is None:
            raise ValueError("Product index not built. Call build_product_index first.")

        # Convert OpenCV image to PIL
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Get image embedding
        embedding = self._get_image_embedding(image_pil)
        
        # Normalize embedding
        faiss.normalize_L2(embedding)
        
        # Search index
        similarities, indices = self.index.search(embedding, 1)
        similarity = float(similarities[0][0])
        
        # Determine match type
        if similarity >= SIMILARITY_THRESHOLDS["exact_match"]:
            match_type = "exact"
        elif similarity >= SIMILARITY_THRESHOLDS["similar_match"]:
            match_type = "similar"
        else:
            match_type = "no_match"
            
        return {
            "match_type": match_type,
            "matched_product_id": self.product_ids[indices[0][0]] if match_type != "no_match" else None,
            "confidence": similarity
        }

    def _get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Get CLIP embedding for an image."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        return image_features.detach().cpu().numpy()

    def _load_image_from_url(self, url: str) -> Optional[Image.Image]:
        """Load image from URL and return PIL Image object."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"Error loading image from {url}: {str(e)}")
            return None 