from pathlib import Path

# Project paths

DATA_DIR = Path("D:/FlickedHack/data/videos")
MODELS_DIR = Path("D:/FlickedHack/models")
CACHE_DIR = Path("D:/FlickedHack/cache")

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, CACHE_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# YOLO Configuration
YOLO_MODEL_NAME = "yolov8n.pt"  #will upgrade agar lagega toh
FASHION_CLASSES = [
    "top", "bottom", "dress", "jacket",
    "earring", "bag", "shoe", "accessory"
]

# CLIP Configuration
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SIMILARITY_THRESHOLDS = {
    "exact_match": 0.15,
    "similar_match": 0.10,
    "no_match": 0.0
}

# Video Processing
FRAME_EXTRACTION_INTERVAL = 1  # Extract 1 frame per second

# Vibe Classification
VIBE_CATEGORIES = [
    "Coquette",
    "Clean Girl",
    "Cottagecore",
    "Streetcore",
    "Y2K",
    "Boho",
    "Party Glam"
]

# API Configuration
DEFAULT_CONFIDENCE_THRESHOLD = 0.5 