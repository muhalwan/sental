import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
from data_preprocessing import prepare_dataloaders
from torch.amp import autocast, GradScaler
from tqdm import tqdm

MODEL_NAME = 'ProsusAI/finbert'
NUM_CLASSES = 3
EPOCHS = 4
BATCH_SIZE = 16
ACCUMULATION_STEPS = 4
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
OUTPUT_DIR = 'model_outputs'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, data_loader, optimizer, scheduler, scaler, accumulation_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Training")
    for i, batch in progress_bar:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        with autocast(device_type='cuda'):
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss = loss / accumulation_steps

        total_loss += outputs.loss.item()
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({'loss': outputs.loss.item()})

    return total_loss / len(data_loader)


def eval_model(model, data_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    return true_labels, predictions


def plot_visualizations(true_labels, predictions, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = classification_report(true_labels, predictions, target_names=['Bearish', 'Bullish', 'Neutral'], output_dict=True)
    print("Classification Report:\n", classification_report(true_labels, predictions, target_names=['Bearish', 'Bullish', 'Neutral']))

    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bearish', 'Bullish', 'Neutral'], yticklabels=['Bearish', 'Bullish', 'Neutral'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()


def generate_wordclouds(texts, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame({'text': texts, 'label': labels})
    for sentiment, name in zip([0, 1, 2], ['Bearish', 'Bullish', 'Neutral']):
        subset_text = ' '.join(df[df['label'] == sentiment]['text'])
        if not subset_text:
            continue
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(subset_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {name} Sentiment')
        plt.savefig(os.path.join(output_dir, f'wordcloud_{name}.png'))
        plt.close()


if __name__ == '__main__':
    base = os.path.dirname(__file__)
    train_csv = os.path.join(base, 'dataset', 'sent_train.csv')
    valid_csv = os.path.join(base, 'dataset', 'sent_valid.csv')

    train_dl, valid_dl, test_dl = prepare_dataloaders(
        train_csv, valid_csv, MODEL_NAME, BATCH_SIZE, MAX_LENGTH
    )

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES).to(DEVICE)
    model.gradient_checkpointing_enable()

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = (len(train_dl) // ACCUMULATION_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scaler = GradScaler()

    torch.cuda.empty_cache()

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        avg_loss = train_epoch(model, train_dl, optimizer, scheduler, scaler, ACCUMULATION_STEPS)
        print(f'Training loss: {avg_loss}')

        true_labels, predictions = eval_model(model, valid_dl)
        accuracy = accuracy_score(true_labels, predictions)
        print(f'Validation Accuracy: {accuracy}')

    print("\nEvaluating on test set...")
    true_labels, predictions = eval_model(model, test_dl)
    plot_visualizations(true_labels, predictions, OUTPUT_DIR)

    test_texts = [test_dl.dataset.texts.iloc[i] for i in range(len(test_dl.dataset))]
    generate_wordclouds(test_texts, predictions, OUTPUT_DIR)

    model.save_pretrained(os.path.join(OUTPUT_DIR, 'fine_tuned_model'))
    test_dl.dataset.tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, 'fine_tuned_model'))

    print(f"\nModel and visualizations saved in '{OUTPUT_DIR}' directory.")
