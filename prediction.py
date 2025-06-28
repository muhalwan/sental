import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

model_path = './model_outputs/fine_tuned_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

while True:
    tweet = input("Please enter a tweet to analyze (or type 'exit' to quit): ")

    if tweet.lower() == 'exit':
        break

    inputs = tokenizer(tweet, return_tensors='pt', truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=-1)

    predicted_class_idx = torch.argmax(probabilities, dim=-1).item()

    predicted_probability = probabilities[0][predicted_class_idx].item()

    labels = ['Bearish', 'Bullish', 'Neutral']
    predicted_label = labels[predicted_class_idx]

    print(f'Tweet: "{tweet}"')
    print(f'Predicted Sentiment: {predicted_label}')
    print(f'Confidence: {predicted_probability:.2%}')
    print("-"*20)
