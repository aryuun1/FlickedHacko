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
import os
import pandas as pd


class ProductMatcher:
    def __init__(self):
        """Initialize CLIP model and load product catalog."""
        print("Initializing CLIP model...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.index = None
        self.product_ids = None
        
        # Check if saved index exists
        if os.path.exists("data/product_index.faiss") and os.path.exists("data/product_ids.npy"):
            print("Loading pre-built product index...")
            self.load_index()
        else:
            print("Building product index for the first time...")
            self.build_product_index()
            self.save_index()
    
    def save_index(self):
        """Save FAISS index and product IDs to disk."""
        print("\nSaving product index...")
        faiss.write_index(self.index, "data/product_index.faiss")
        np.save("data/product_ids.npy", self.product_ids)
        print("Product index saved successfully")
    
    def load_index(self):
        """Load FAISS index and product IDs from disk."""
        self.index = faiss.read_index("data/product_index.faiss")
        self.product_ids = np.load("data/product_ids.npy")
        print(f"Loaded index with {self.index.ntotal} products")
    
    def build_product_index(self):
        """Build FAISS index from first 1000 products in catalog as training set."""
        # Read product catalog
        df = pd.read_csv("data/images.csv")
        
        # Take only first 1000 products
        df = df.head(1000)
        print(f"\nBuilding index with first {len(df)} products as training set...")
        
        # Initialize FAISS index for cosine similarity
        embedding_size = 512  # CLIP's image embedding size
        self.index = faiss.IndexFlatIP(embedding_size)  # Inner product = cosine similarity for normalized vectors
        self.product_ids = []
        
        # Process each product
        successful_products = 0
        for idx, row in df.iterrows():
            print(f"\nProcessing product {idx+1}/{len(df)}: {row['id']}")
            
            # Download and process image
            print(f"URL: {row['image_url']}")
            try:
                print("Fetching image from URL:", row['image_url'])
                response = requests.get(row['image_url'], timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                print("Image loaded successfully")
                
                # Get image embedding
                inputs = self.processor(images=image, return_tensors="pt")
                image_features = self.model.get_image_features(**inputs)
                embedding = image_features.detach().numpy()
                
                # Normalize embedding for cosine similarity
                faiss.normalize_L2(embedding)
                
                # Add to index
                self.index.add(embedding)
                self.product_ids.append(str(row['id']))
                successful_products += 1
                
                print(f"Successfully processed product {row['id']}")
            except Exception as e:
                print(f"Error processing product {row['id']}: {str(e)}")
                continue
        
        print(f"\nFinished building index with {successful_products} successfully processed products")
    
    def match_product(self, image: np.ndarray) -> Dict[str, any]:
        """Match image with products in catalog."""
        if self.index is None:
            raise ValueError("Product index not built. Call build_product_index first.")
        
        # Convert image to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Get image embedding
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model.get_image_features(**inputs)
        query_embedding = image_features.detach().numpy()
        
        # Normalize query embedding for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search index (cosine similarity is inner product of normalized vectors)
        similarities, indices = self.index.search(query_embedding, 1)
        similarity = float(similarities[0][0])  # Already cosine similarity
        
        print(f"\nTop match similarity score: {similarity:.4f}")
        
        # Determine match type based on similarity
        if similarity >= 0.7:  # Adjusted for cosine similarity
            match_type = "exact"
            print(f"Found exact match: Product {self.product_ids[indices[0][0]]} with similarity {similarity}")
        elif similarity >= 0.5:  # Adjusted for cosine similarity
            match_type = "similar"
            print(f"Found similar match: Product {self.product_ids[indices[0][0]]} with similarity {similarity}")
        else:
            match_type = "no_match"
            print("No good match found")
        
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
            print(f"Fetching image from URL: {url}")
            response = requests.get(url, timeout=10)  # Add timeout
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            print("Image loaded successfully")
            return image
        except Exception as e:
            print(f"Error loading image from {url}: {str(e)}")
            return None 