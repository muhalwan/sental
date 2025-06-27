import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
from data_preprocessing import prepare_dataloaders
from torch.amp import autocast, GradScaler
from tqdm import tqdm


MODEL_NAME = 'ProsusAI/finbert'
NUM_CLASSES = 3
EPOCHS = 10
BATCH_SIZE = 16
ACCUMULATION_STEPS = 4
MAX_LENGTH = 256
LEARNING_RATE = 3e-5
OUTPUT_DIR = 'model_outputs'
EARLY_STOPPING_PATIENCE = 3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, scaler, accumulation_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Training")
    for i, batch in progress_bar:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        with autocast(device_type='cuda'):
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss = loss / accumulation_steps

        total_loss += loss.item() * accumulation_steps
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})

    return total_loss / len(data_loader.dataset)


def eval_model(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            with autocast(device_type='cuda'):
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    return true_labels, predictions, total_loss / len(data_loader)


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

    train_dl, valid_dl, test_dl, class_weights = prepare_dataloaders(
        train_csv, valid_csv, MODEL_NAME, BATCH_SIZE, MAX_LENGTH
    )

    class_weights = class_weights.to(DEVICE)
    loss_fn = CrossEntropyLoss(weight=class_weights)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES).to(DEVICE)
    model.gradient_checkpointing_enable()

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = (len(train_dl) // ACCUMULATION_STEPS) * EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler()

    torch.cuda.empty_cache()

    best_accuracy = 0
    epochs_no_improve = 0

    start_time = time.time()
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        avg_train_loss = train_epoch(model, train_dl, loss_fn, optimizer, scheduler, scaler, ACCUMULATION_STEPS)
        print(f'Training loss: {avg_train_loss}')

        true_labels, predictions, avg_val_loss = eval_model(model, valid_dl, loss_fn)
        accuracy = accuracy_score(true_labels, predictions)
        elapsed_time = time.time() - start_time
        print(f'Validation Accuracy: {accuracy:.4f}, Validation Loss: {avg_val_loss:.4f} | Time: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save_pretrained(os.path.join(OUTPUT_DIR, 'fine_tuned_model'))
            test_dl.dataset.tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, 'fine_tuned_model'))
            epochs_no_improve = 0
            print("New best model saved.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")

    print("\nLoading best model for final evaluation on test set...")
    model = BertForSequenceClassification.from_pretrained(os.path.join(OUTPUT_DIR, 'fine_tuned_model'), use_safetensors=True).to(DEVICE)

    true_labels, predictions, _ = eval_model(model, test_dl, loss_fn)
    plot_visualizations(true_labels, predictions, OUTPUT_DIR)

    test_texts = [test_dl.dataset.texts.iloc[i] for i in range(len(test_dl.dataset))]
    generate_wordclouds(test_texts, predictions, OUTPUT_DIR)

    print(f"\nModel and visualizations saved in '{OUTPUT_DIR}' directory.")
