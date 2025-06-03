from src import FlickdEngine
import json

def main():
    # Initialize the Flickd engine
    engine = FlickdEngine()

    # Example product catalog (you would need to replace with actual product data)
    product_catalog = [
        {
            "id": "prod_001",
            "image_url": "https://example.com/product1.jpg"
        },
        # Add more products...
    ]

    # Build product index
    engine.product_matcher.build_product_index(product_catalog)

    # Process a video
    video_path = "path/to/your/video.mp4"
    caption = "Summer vibes with my favorite cottagecore dress! 🌸 #cottagecore #vintage #floral"
    hashtags = ["cottagecore", "vintage", "floral"]

    # Process the video and get results
    results = engine.process_video(
        video_path=video_path,
        caption=caption,
        hashtags=hashtags
    )

    # Save results
    engine.save_results(results, "output.json")

    # Print results
    print("\nProcessing complete! Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main() 