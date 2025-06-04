from src import FlickdEngine
import json
from pathlib import Path
import argparse
from typing import List, Dict
import csv
import random
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Process video with Flickd Engine')
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    return parser.parse_args()

def load_product_catalog(csv_path: str, max_products: int = 200) -> List[Dict]:
    """
    Load product catalog from CSV file.
    Args:
        csv_path: Path to the CSV file
        max_products: Maximum number of products to load (default: 200)
    """
    products = {}
    print(f"\nReading CSV file: {csv_path}")
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                product_id = row['id']
                image_url = row['image_url'].strip()
                
                # Skip if URL is empty
                if not image_url:
                    continue
                    
                if product_id not in products:
                    print(f"Adding product {product_id} with URL: {image_url}")
                    products[product_id] = {
                        "id": product_id,
                        "image_url": image_url,
                        "type": "fashion_item",  # You can enhance this with actual product types
                        "color": random.choice(["black", "white", "blue", "red", "green"])  # Placeholder colors
                    }
                    # Break if we've reached the maximum number of products
                    if len(products) >= max_products:
                        break
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
    
    product_list = list(products.values())
    print(f"\nSelected {len(product_list)} products from catalog")
    
    # Verify the first few products
    print("\nFirst few products:")
    for i, product in enumerate(product_list[:3]):
        print(f"{i+1}. ID: {product['id']}, URL: {product['image_url']}")
    
    return product_list

def process_media(media_path: str, caption: str, hashtags: List[str], engine: FlickdEngine) -> Dict:
    """Process either an image or video file."""
    # Determine if input is image or video based on extension
    is_video = media_path.lower().endswith(('.mp4', '.avi', '.mov', '.wmv'))
    
    if is_video:
        return engine.process_video(
            video_path=media_path,
            caption=caption,
            hashtags=hashtags
        )
    else:
        return engine.process_image(
            image_path=media_path,
            caption=caption,
            hashtags=hashtags
        )

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize FlickdEngine
    engine = FlickdEngine()
    
    # Process video
    print(f"\nProcessing video: {args.input}")
    
    # Get video metadata
    video_path = args.input
    video_name = os.path.basename(video_path)
    video_id = video_name.split('.')[0]  # Use filename without extension as ID
    
    # Process the video
    results = engine.process_video(video_path)
    
    # Save results
    output_path = os.path.join(f'outputhaha_{video_id}.json')
    engine.save_results(results, output_path)
    print(f"\nResults saved to: {output_path}")

if __name__ == '__main__':
    main() 