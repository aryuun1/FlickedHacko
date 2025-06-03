from src import FlickdEngine
import json
from pathlib import Path

def main():
    # Initialize the Flickd engine
    engine = FlickdEngine()

    # Sample product catalog with some example fashion items
    product_catalog = [
        {
            "id": "dress_001",
            "image_url": "https://raw.githubusercontent.com/openai/CLIP/main/CLIP.png"  # Using CLIP's image as placeholder
        }
    ]

    # Build product index
    print("Building product index...")
    engine.product_matcher.build_product_index(product_catalog)

    # Get the path to the test video
    data_dir = Path("data")
    videos_dir = data_dir / "videos"
    videos_dir.mkdir(exist_ok=True, parents=True)
    
    # Using the actual video file name
    video_path = videos_dir / "2025-05-27_13-46-16_UTC.mp4"
                               
    if not video_path.exists():
        print(f"\nPlease place a test video file at: {video_path}")
        return

    # Sample video metadata
    caption = "Sunkissed Summer has landed. Think golden hour, every hour. Easy silhouettes. Bare shoulders. Dresses that breathe. Made for holiday glow, even if you’re just stepping out for coffee."
    hashtags = ["cottagecore", "vintage", "floral", "summer", "dress"]

    print(f"\nProcessing video: {video_path}")
    print("Caption:", caption)
    print("Hashtags:", hashtags)

    # Process the video and get results
    results = engine.process_video(
        video_path=str(video_path),
        caption=caption,
        hashtags=hashtags
    )

    # Save results
    output_path = f"outputs/output_{video_path.stem}.json"
    engine.save_results(results, output_path)

    # Print results
    print(f"\nProcessing complete! Results saved to: {output_path}")
    print("\nResults preview:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main() 