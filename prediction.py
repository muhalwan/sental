import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os
from huggingface_hub import snapshot_download

# The local path where the model will be stored
local_model_dir = './model_outputs/fine_tuned_model'

# Get the model repo ID from the environment variable
MODEL_REPO_ID = os.getenv('MODEL_URL')

# Use the local directory as the default model path
model_path = local_model_dir

# Download the model from the repo if a key file doesn't exist locally
if not os.path.exists(os.path.join(local_model_dir, "model.safetensors")):
    if MODEL_REPO_ID:
        print(f"Model not found locally. Downloading from {MODEL_REPO_ID}...")
        # Use snapshot_download and get the actual path, disabling symlinks for compatibility
        model_path = snapshot_download(
            repo_id=MODEL_REPO_ID,
            local_dir=local_model_dir,
            repo_type='model',
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded successfully to {model_path}.")
    else:
        raise ValueError("Model not found locally and MODEL_URL environment variable is not set.")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=-1).squeeze()

    predicted_class_idx = torch.argmax(probabilities).item()
    labels = ['Bearish', 'Bullish', 'Neutral']
    prediction = labels[predicted_class_idx]

    probs_dict = {label: prob.item() for label, prob in zip(labels, probabilities)}

    return prediction, probs_dict

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
