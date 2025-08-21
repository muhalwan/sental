import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForSequenceClassification
import torch.nn.functional as F
import os
import logging
import time
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

local_model_dir = './model_outputs/fine_tuned_model'

MODEL_REPO_ID = os.getenv('MODEL_URL')
if MODEL_REPO_ID:
    MODEL_REPO_ID = MODEL_REPO_ID.strip()
    if not MODEL_REPO_ID:
        MODEL_REPO_ID = None

model_path = local_model_dir


def load_model_with_retry(max_retries: int = 3, retry_delay: float = 2.0):
    global model_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(os.path.join(local_model_dir, "adapter_model.safetensors")):
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

    for attempt in range(max_retries):
        try:
            logger.info(f"Loading tokenizer from {model_path}...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            logger.info(f"Loading model from {model_path}...")
            torch_dtype = torch.float16 if device.type == 'cuda' else torch.float32
            model = AutoPeftModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map='auto'
            )
            model = model.merge_and_unload()

            logger.info("Model and tokenizer loaded successfully.")
            return tokenizer, model

        except Exception as e:
            logger.warning(f"Model loading attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Failed to load model after {max_retries} attempts: {e}")


try:
    tokenizer, model = load_model_with_retry()
except Exception as e:
    logger.error(f"Critical error during model initialization: {e}")
    raise


def predict_sentiment(text: str, max_length: int = 512):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")

    if not text or len(text.strip()) == 0:
        raise ValueError("Input text cannot be empty")

    if len(text) > max_length * 10:
        text = text[:max_length * 10]

    try:
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(model.device)

        with torch.inference_mode():
            outputs = model(**inputs)

        probabilities = F.softmax(outputs.logits, dim=-1).squeeze()
        predicted_class_idx = torch.argmax(probabilities).item()

        labels = ['Bearish', 'Bullish', 'Neutral']
        prediction = labels[predicted_class_idx]

        probs_dict = {label: prob.item() for label, prob in zip(labels, probabilities)}

        return prediction, probs_dict

    except Exception as e:
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
        print("-" * 20)