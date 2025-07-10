import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from wordcloud import WordCloud
from data_preprocessing import prepare_dataloaders
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import torch.nn.functional as F
import optuna

MODEL_NAME = 'ProsusAI/finbert'
NUM_CLASSES = 3
EPOCHS = 10
BATCH_SIZE = 16
ACCUMULATION_STEPS = 8
MAX_LENGTH = 256
LEARNING_RATE = 3e-5
OUTPUT_DIR = 'model_outputs'
EARLY_STOPPING_PATIENCE = 3
N_TRIALS = 15

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
    predictions, true_labels, all_probs = [], [], []
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
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    return true_labels, predictions, np.array(all_probs), total_loss / len(data_loader)


def objective(trial, train_csv, valid_csv):
    LEARNING_RATE = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    BATCH_SIZE = trial.suggest_categorical("batch_size", [4, 8])

    train_dl, valid_dl, _, class_weights = prepare_dataloaders(
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
    for epoch in range(EPOCHS):
        train_epoch(model, train_dl, loss_fn, optimizer, scheduler, scaler, ACCUMULATION_STEPS)
        true_labels, predictions, _, _ = eval_model(model, valid_dl, loss_fn)
        accuracy = accuracy_score(true_labels, predictions)

        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if accuracy > best_accuracy:
            best_accuracy = accuracy

        if epoch > EARLY_STOPPING_PATIENCE and accuracy < best_accuracy:
             break

    del model, optimizer, scheduler, scaler, train_dl, valid_dl, class_weights
    torch.cuda.empty_cache()

    return best_accuracy


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


def plot_roc_curves(true_labels, predictions_prob, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    true_labels_bin = label_binarize(true_labels, classes=[0, 1, 2])
    n_classes = true_labels_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], predictions_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    class_names = ['Bearish', 'Bullish', 'Neutral']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
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


def visualize_attention(model, tokenizer, sentences, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    for i, sentence in enumerate(sentences):
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=MAX_LENGTH).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        attentions = outputs.attentions[-1]
        attention = torch.mean(attentions, dim=1).squeeze(0)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        plt.figure(figsize=(12, 10))
        sns.heatmap(attention.cpu().numpy(), xticklabels=tokens, yticklabels=tokens, cmap='viridis')
        plt.title(f'Attention Heatmap for: "{sentence[:50]}..."')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.savefig(os.path.join(output_dir, f'attention_heatmap_{i+1}.png'), bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    base = os.path.dirname(__file__)
    train_csv = os.path.join(base, 'dataset', 'sent_train.csv')
    valid_csv = os.path.join(base, 'dataset', 'sent_valid.csv')

    print("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, train_csv, valid_csv), n_trials=N_TRIALS)

    print("\nOptimization finished.")
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (Validation Accuracy): {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    print("\nStarting final training with the best hyperparameters...")
    best_params = best_trial.params
    LEARNING_RATE = best_params['lr']
    BATCH_SIZE = best_params['batch_size']

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

        true_labels, predictions, _, avg_val_loss = eval_model(model, valid_dl, loss_fn)
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
    model = BertForSequenceClassification.from_pretrained(os.path.join(OUTPUT_DIR, 'fine_tuned_model')).to(DEVICE)

    model.config.to_json_file(os.path.join(OUTPUT_DIR, 'config.json'))
    model.save_pretrained(OUTPUT_DIR)

    print("\nConverting model to ONNX format...")
    onnx_path = os.path.join(OUTPUT_DIR, 'model.onnx')
    dummy_input = torch.zeros(1, MAX_LENGTH, dtype=torch.long).to(DEVICE)
    dummy_attention_mask = torch.zeros(1, MAX_LENGTH, dtype=torch.long).to(DEVICE)

    torch.onnx.export(
        model,
        (dummy_input, dummy_attention_mask),
        onnx_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size'}
        },
        opset_version=14,
        export_params=True
    )
    print(f"ONNX model saved to {onnx_path}")

    true_labels, predictions, probs, _ = eval_model(model, test_dl, loss_fn)
    plot_visualizations(true_labels, predictions, OUTPUT_DIR)
    plot_roc_curves(true_labels, probs, OUTPUT_DIR)

    test_texts = [test_dl.dataset.texts.iloc[i] for i in range(len(test_dl.dataset))]
    generate_wordclouds(test_texts, predictions, OUTPUT_DIR)

    print("\nGenerating attention visualizations for sample texts...")
    sample_texts = [
        "BTIG points to breakfast pressure for Dunkin' Brands",
        "$CX - Cemex cut at Credit Suisse, J.P. Morgan on weak building outlook",
        "Adobe price target raised to $350 vs. $320 at Canaccord"
    ]
    visualize_attention(model, test_dl.dataset.tokenizer, sample_texts, OUTPUT_DIR)

    print(f"\nModel and visualizations saved in '{OUTPUT_DIR}' directory.")
