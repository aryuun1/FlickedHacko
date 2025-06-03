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
        print("Initializing CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.index = None
        self.product_ids = []
        self.product_details = []

    def build_product_index(self, product_images: List[Dict[str, str]]):
        """
        Build FAISS index from product catalog images.
        product_images: List of dicts with 'id' and 'image_url' keys
        """
        print(f"\nBuilding product index with {len(product_images)} products...")
        embeddings = []
        self.product_ids = []
        self.product_details = []

        for i, product in enumerate(product_images, 1):
            print(f"\nProcessing product {i}/{len(product_images)}: {product['id']}")
            print(f"URL: {product['image_url']}")
            
            # Load and process image
            try:
                image = self._load_image_from_url(product['image_url'])
                if image is None:
                    print(f"Failed to load image for product {product['id']}")
                    continue

                # Get image embedding
                embedding = self._get_image_embedding(image)
                embeddings.append(embedding)
                self.product_ids.append(product['id'])
                self.product_details.append(product)
                print(f"Successfully processed product {product['id']}")
            except Exception as e:
                print(f"Error processing product {product['id']}: {str(e)}")
                continue

        if not embeddings:
            raise ValueError("No valid product embeddings generated!")

        # Convert to numpy array and normalize
        print("\nFinalizing index...")
        embeddings = np.vstack(embeddings)
        faiss.normalize_L2(embeddings)

        # Build FAISS index
        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product = cosine similarity for normalized vectors
        self.index.add(embeddings)
        print(f"Index built successfully with {len(self.product_ids)} products")

    def match_product(self, image: np.ndarray) -> Dict[str, any]:
        """
        Match detected product against catalog.
        Returns match type and product ID.
        """
        if self.index is None:
            raise ValueError("Product index not built. Call build_product_index first.")

        try:
            # Convert OpenCV image to PIL
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Get image embedding
            embedding = self._get_image_embedding(image_pil)
            
            # Normalize embedding
            faiss.normalize_L2(embedding)
            
            # Search index
            similarities, indices = self.index.search(embedding, 3)  # Get top 3 matches for better chances
            similarity = float(similarities[0][0])
            
            print(f"\nTop match similarity score: {similarity:.4f}")
            
            # Lower thresholds to increase matches
            if similarity >= 0.15:  # Lowered from 0.25
                match_type = "exact"
            elif similarity >= 0.10:  # Lowered from 0.15
                match_type = "similar"
            else:
                match_type = "no_match"
            
            matched_product = None
            if match_type != "no_match":
                product_idx = indices[0][0]
                matched_product = self.product_details[product_idx]
                print(f"Found {match_type} match: Product {matched_product['id']} with similarity {similarity:.4f}")
            else:
                print("No match found")
                
            return {
                "match_type": match_type,
                "matched_product_id": self.product_ids[indices[0][0]] if match_type != "no_match" else None,
                "confidence": similarity,
                "color": matched_product.get("color") if matched_product else None,
                "product_type": matched_product.get("type") if matched_product else None
            }
        except Exception as e:
            print(f"Error in product matching: {str(e)}")
            return {
                "match_type": "error",
                "matched_product_id": None,
                "confidence": 0.0,
                "error": str(e)
            }

    def _get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Get CLIP embedding for an image."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        return image_features.detach().cpu().numpy()

    def _load_image_from_url(self, url: str) -> Optional[Image.Image]:
        """Load image from URL and return PIL Image object."""
        try:
            print(f"Fetching image from URL: {url}")
            response = requests.get(url, timeout=10)  # Add timeout
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            print("Image loaded successfully")
            return image
        except Exception as e:
            print(f"Error loading image from {url}: {str(e)}")
            return None 