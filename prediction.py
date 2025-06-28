import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os
from huggingface_hub import snapshot_download

model_path = './model_outputs/fine_tuned_model'

MODEL_URL = os.getenv('MODEL_URL')

if not os.path.exists(model_path) and MODEL_URL:
    print(f"Model not found locally. Downloading from {MODEL_URL}...")
    snapshot_download(repo_id=MODEL_URL, local_dir=model_path, repo_type='model')
    print("Model downloaded successfully.")
elif not os.path.exists(model_path):
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
