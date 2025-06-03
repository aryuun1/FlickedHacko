# AI Vision Project

This project combines multiple AI vision and NLP technologies including YOLO, CLIP, FAISS, and SpaCy along with Hugging Face tools.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Unix/MacOS:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download SpaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Dependencies

- YOLO (ultralytics) - Object detection
- CLIP - Image-text understanding
- FAISS - Similarity search and clustering
- SpaCy - Natural Language Processing
- Hugging Face Transformers - State-of-the-art ML models
- PyTorch - Deep learning framework

## Project Structure

```
.
├── data/           # Dataset storage
├── models/         # Saved models
├── notebooks/      # Jupyter notebooks
└── src/           # Source code
```

## Getting Started

After setting up the environment, you can start using the various AI components:

- YOLO for object detection
- CLIP for image-text matching
- FAISS for efficient similarity search
- SpaCy for text processing
- Hugging Face models for various ML tasks

## Note

Make sure you have sufficient disk space and computational resources as some models might be large. 