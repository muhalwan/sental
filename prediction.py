import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os
import logging
import time
from typing import Tuple, Dict, Optional
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

local_model_dir = './model_outputs/fine_tuned_model'

# Validate and get MODEL_REPO_ID
MODEL_REPO_ID = os.getenv('MODEL_URL')
if MODEL_REPO_ID:
    MODEL_REPO_ID = MODEL_REPO_ID.strip()
    if not MODEL_REPO_ID:
        MODEL_REPO_ID = None

model_path = local_model_dir

def load_model_with_retry(max_retries: int = 3, retry_delay: float = 2.0) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """
    Load model and tokenizer with retry logic and error handling.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        retry_delay (float): Delay between retries in seconds
        
    Returns:
        Tuple[AutoTokenizer, AutoModelForSequenceClassification]: Loaded tokenizer and model
        
    Raises:
        RuntimeError: If model loading fails after all retries
    """
    global model_path
    
    # Check if model exists locally
    if not os.path.exists(os.path.join(local_model_dir, "model.safetensors")):
        if MODEL_REPO_ID:
            logger.info(f"Model not found locally. Downloading from {MODEL_REPO_ID}...")
            for attempt in range(max_retries):
                try:
                    model_path = snapshot_download(
                        repo_id=MODEL_REPO_ID,
                        local_dir=local_model_dir,
                        repo_type='model',
                        local_dir_use_symlinks=False
                    )
                    logger.info(f"Model downloaded successfully to {model_path}.")
                    break
                except Exception as e:
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        raise RuntimeError(f"Failed to download model after {max_retries} attempts: {e}")
        else:
            raise ValueError("Model not found locally and MODEL_URL environment variable is not set.")
    
    # Load tokenizer and model with error handling
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading tokenizer from {model_path}...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            logger.info(f"Loading model from {model_path}...")
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            logger.info("Model and tokenizer loaded successfully.")
            return tokenizer, model
            
        except Exception as e:
            logger.warning(f"Model loading attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Failed to load model after {max_retries} attempts: {e}")

# Load model and tokenizer
try:
    tokenizer, model = load_model_with_retry()
except Exception as e:
    logger.error(f"Critical error during model initialization: {e}")
    raise

def predict_sentiment(text: str, max_length: int = 512) -> Tuple[str, Dict[str, float]]:
    """
    Predict sentiment for given text with comprehensive error handling.
    
    Args:
        text (str): Input text for sentiment analysis
        max_length (int): Maximum token length for input text
        
    Returns:
        Tuple[str, Dict[str, float]]: Predicted sentiment label and probability dict
        
    Raises:
        ValueError: If input text is invalid
        RuntimeError: If prediction fails
    """
    # Input validation
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")
    
    if not text or len(text.strip()) == 0:
        raise ValueError("Input text cannot be empty")
    
    if len(text) > max_length * 10:  # Rough character limit (10 chars per token avg)
        logger.warning(f"Input text is very long ({len(text)} chars), truncating...")
        text = text[:max_length * 10]
    
    try:
        # Tokenization with error handling
        logger.debug(f"Tokenizing text of length {len(text)}")
        inputs = tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=max_length,
            padding=True
        )
        
        # Prediction with error handling
        logger.debug("Running model inference")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process outputs
        probabilities = F.softmax(outputs.logits, dim=-1).squeeze()
        predicted_class_idx = torch.argmax(probabilities).item()
        
        # Define labels (consistent with training)
        labels = ['Bearish', 'Bullish', 'Neutral']
        prediction = labels[predicted_class_idx]
        
        # Convert to dictionary
        probs_dict = {label: prob.item() for label, prob in zip(labels, probabilities)}
        
        logger.debug(f"Prediction: {prediction}, Confidence: {probs_dict[prediction]:.3f}")
        return prediction, probs_dict
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise RuntimeError(f"Failed to predict sentiment: {str(e)}")

if __name__ == '__main__':
    while True:
        tweet = input("Please enter a tweet to analyze (or type 'exit' to quit): ")

        if tweet.lower() == 'exit':
            break

        prediction, probabilities = predict_sentiment(tweet)

        print(f'\nTweet: "{tweet}"')
        print(f'Predicted Sentiment: {prediction}')
        print('Probabilities:')
        for sentiment, prob in probabilities.items():
            print(f'  - {sentiment}: {prob:.2%}')
        print("-"*20)
