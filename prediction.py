import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

model_path = './model_outputs/fine_tuned_model'
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
