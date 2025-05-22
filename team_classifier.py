"""
Team classification module using SigLIP model
"""

import numpy as np
import torch
from transformers import AutoProcessor, SiglipVisionModel
from sklearn.cluster import KMeans


class TeamClassifier:
    """Classify players into teams using image embedding and clustering."""
    
    def __init__(self, model_path="google/siglip-base-patch16-224", token=None, device=None):
        """
        Initialize the team classifier.
        
        Args:
            model_path: Path or name of the SigLIP model
            token: Hugging Face token for accessing models
            device: Device to run the model on (cuda or cpu)
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_path, token=token)
        self.model = SiglipVisionModel.from_pretrained(model_path, token=token).to(self.device)
        
        # Initialize classifier to None (will be trained on first batch)
        self.classifier = None
        
    def extract_features(self, images):
        """Extract features from images using SigLIP."""
        if not images:
            return []
        
        with torch.no_grad():
            # Process images
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs)
            
            # Get embeddings (using pooler output)
            embeddings = outputs.pooler_output.cpu().numpy()
            
        return embeddings
    
    def train(self, images):
        """Train the classifier on a batch of images."""
        if not images:
            print("No images provided for training")
            return
        
        # Extract features
        features = self.extract_features(images)
        
        # Use K-means clustering to separate teams (k=2)
        self.classifier = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.classifier.fit(features)
        
        print(f"Team classifier trained on {len(images)} images")
    
    def predict(self, images):
        """Predict team for each image."""
        if not images:
            return []
        
        # If classifier not trained yet, train it first
        if self.classifier is None:
            self.train(images)
        
        # Extract features
        features = self.extract_features(images)
        
        # Predict teams
        predictions = self.classifier.predict(features)
        
        return predictions
    
    def predict_with_score(self, images):
        """Predict team with confidence score."""
        if not images:
            return [], []
        
        # If classifier not trained yet, train it first
        if self.classifier is None:
            self.train(images)
        
        # Extract features
        features = self.extract_features(images)
        
        # Predict teams
        predictions = self.classifier.predict(features)
        
        # Calculate distance to cluster centers
        distances = self.classifier.transform(features)
        
        # Calculate confidence (1 - normalized distance)
        confidences = 1.0 - distances[np.arange(len(predictions)), predictions] / np.max(distances)
        
        return predictions, confidences
