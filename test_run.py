from src import FlickdEngine
import json
from pathlib import Path
import argparse
from typing import List, Dict
import csv
import random

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
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process video or image with Flickd Engine')
    parser.add_argument('--input', type=str, help='Path to input video or image file')
    parser.add_argument('--caption', type=str, default="Sunkissed Summer has landed. Think golden hour, every hour. Easy silhouettes. Bare shoulders. Dresses that breathe. Made for holiday glow, even if you're just stepping out for coffee.", help='Caption text')
    parser.add_argument('--hashtags', type=str, nargs='+', default=["summer", "fashion", "style"], help='List of hashtags')
    parser.add_argument('--max-products', type=int, default=200, help='Maximum number of products to load from catalog')
    args = parser.parse_args()

    # Initialize the Flickd engine
    engine = FlickdEngine()

    # Load product catalog from CSV
    data_dir = Path("data")
    csv_path = data_dir / "images.csv"
    
    if not csv_path.exists():
        print(f"\nError: Product catalog CSV not found at: {csv_path}")
        return
        
    print("Loading product catalog from CSV...")
    product_catalog = load_product_catalog(str(csv_path), args.max_products)
    print(f"Loaded {len(product_catalog)} unique products")

    # Build product index
    print("\nBuilding product index...")
    engine.product_matcher.build_product_index(product_catalog)

    # Get the path to the input file
    if args.input:
        media_path = Path(args.input)
    else:
        # Default to the test video if no input specified
        videos_dir = data_dir / "videos"
        videos_dir.mkdir(exist_ok=True, parents=True)
        media_path = videos_dir / "2025-05-27_13-46-16_UTC.mp4"

    if not media_path.exists():
        print(f"\nError: Input file not found at: {media_path}")
        return

    print(f"\nProcessing: {media_path}")
    print("Caption:", args.caption)
    print("Hashtags:", args.hashtags)

    # Process the media and get results
    results = process_media(
        str(media_path),
        args.caption,
        args.hashtags,
        engine
    )

    # Create outputs directory if it doesn't exist
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    # Save results
    output_path = outputs_dir / f"output_{media_path.stem}.json"
    engine.save_results(results, output_path)

    # Print results
    print(f"\nProcessing complete! Results saved to: {output_path}")
    print("\nResults preview:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main() 