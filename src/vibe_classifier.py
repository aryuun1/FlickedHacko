import spacy
from transformers import pipeline
import numpy as np
from typing import List, Dict
from .config import VIBE_CATEGORIES

class VibeClassifier:
    def __init__(self):
        """Initialize NLP models for vibe classification."""
        # Load spaCy model for text processing
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load sentiment analysis pipeline
        self.sentiment_analyzer = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # CPU
        )

        # Vibe keywords and associations
        self.vibe_keywords = {
            "Coquette": ["cute", "feminine", "soft", "pink", "bows", "lace", "pearls", "ribbons", "dainty"],
            "Clean Girl": ["minimal", "clean", "sleek", "natural", "effortless", "neutral", "basic", "simple"],
            "Cottagecore": ["cottage", "rural", "floral", "vintage", "garden", "prairie", "pastoral", "rustic"],
            "Streetcore": ["urban", "street", "edgy", "cool", "graffiti", "hip hop", "sneakers"],
            "Y2K": ["2000s", "retro", "colorful", "butterfly", "crop top", "low rise", "glitter"],
            "Boho": ["bohemian", "free spirit", "earthy", "natural", "flowy", "ethnic", "hippie"],
            "Party Glam": ["glamorous", "sparkle", "sequin", "party", "night out", "dressy", "elegant"]
        }

    def classify_vibes(self, text: str) -> List[str]:
        """
        Classify text into vibes based on keywords and sentiment.
        Returns list of most likely vibes (1-3 categories).
        """
        # Process text with spaCy
        doc = self.nlp(text.lower())
        
        # Get sentiment
        sentiment = self.sentiment_analyzer(text)[0]
        is_positive = sentiment["label"] == "POSITIVE"
        
        # Calculate vibe scores
        vibe_scores = {}
        for vibe, keywords in self.vibe_keywords.items():
            score = 0
            for keyword in keywords:
                # Check for keyword matches in text
                if keyword in text.lower():
                    score += 1
                
                # Check for semantic similarity with keywords in text
                for token in doc:
                    if token.has_vector and self.nlp(keyword)[0].has_vector:
                        similarity = token.similarity(self.nlp(keyword)[0])
                        if similarity > 0.7:  # Threshold for semantic similarity
                            score += similarity
            
            vibe_scores[vibe] = score

        # Adjust scores based on sentiment
        if is_positive:
            for vibe in ["Party Glam", "Clean Girl", "Coquette"]:
                if vibe in vibe_scores:
                    vibe_scores[vibe] *= 1.2

        # Get top vibes (1-3 categories)
        sorted_vibes = sorted(vibe_scores.items(), key=lambda x: x[1], reverse=True)
        top_vibes = [vibe for vibe, score in sorted_vibes if score > 0][:3]

        return top_vibes if top_vibes else ["Clean Girl"]  # Default to Clean Girl if no strong matches 