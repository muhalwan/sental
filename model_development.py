import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import warnings
warnings.filterwarnings("ignore", message="expandable_segments not supported on this platform")
import torch

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
from sklearn.model_selection import StratifiedKFold
from wordcloud import WordCloud
from data_preprocessing import prepare_dataloaders
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import torch.nn.functional as F
import optuna
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = 'ProsusAI/finbert'
NUM_CLASSES = 3
EPOCHS = 15
BATCH_SIZE = 16
ACCUMULATION_STEPS = 4
MAX_LENGTH = 256
LEARNING_RATE = 3e-5
OUTPUT_DIR = 'model_outputs'
EARLY_STOPPING_PATIENCE = 5
N_TRIALS = 25
USE_CROSS_VALIDATION = True
N_FOLDS = 5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")


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
    # Expanded hyperparameter space
    LEARNING_RATE = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    BATCH_SIZE = trial.suggest_categorical("batch_size", [8, 16, 32])
    ACCUMULATION_STEPS = trial.suggest_categorical("accumulation_steps", [2, 4, 8])
    WEIGHT_DECAY = trial.suggest_float("weight_decay", 0.0, 0.1)
    WARMUP_RATIO = trial.suggest_float("warmup_ratio", 0.05, 0.2)
    DROPOUT_RATE = trial.suggest_float("dropout_rate", 0.1, 0.5)
    AUGMENT_PROB = trial.suggest_float("augment_prob", 0.0, 0.5)
    
    logger.info(f"Trial {trial.number}: LR={LEARNING_RATE:.2e}, BS={BATCH_SIZE}, "
                f"WD={WEIGHT_DECAY:.3f}, Warmup={WARMUP_RATIO:.3f}, Dropout={DROPOUT_RATE:.3f}")

    if USE_CROSS_VALIDATION:
        df_train = pd.read_csv(train_csv)
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, df_train['label'])):
            logger.info(f"  Fold {fold + 1}/{N_FOLDS}")
            
            train_fold_df = df_train.iloc[train_idx].reset_index(drop=True)
            valid_fold_df = df_train.iloc[val_idx].reset_index(drop=True)
            
            train_fold_csv = f'temp_train_fold_{fold}.csv'
            valid_fold_csv = f'temp_valid_fold_{fold}.csv'
            train_fold_df.to_csv(train_fold_csv, index=False)
            valid_fold_df.to_csv(valid_fold_csv, index=False)
            
            try:
                train_dl, valid_dl, _, class_weights = prepare_dataloaders(
                    train_fold_csv, valid_fold_csv, MODEL_NAME, BATCH_SIZE, MAX_LENGTH, 
                    augment_prob=AUGMENT_PROB
                )

                class_weights = class_weights.to(DEVICE)
                loss_fn = CrossEntropyLoss(weight=class_weights)

                model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES).to(DEVICE)
                model.gradient_checkpointing_enable()
                
                if hasattr(model.config, 'hidden_dropout_prob'):
                    model.config.hidden_dropout_prob = DROPOUT_RATE
                    model.config.attention_probs_dropout_prob = DROPOUT_RATE

                optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
                total_steps = (len(train_dl) // ACCUMULATION_STEPS) * EPOCHS
                warmup_steps = int(total_steps * WARMUP_RATIO)
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
                scaler = GradScaler()

                torch.cuda.empty_cache()

                best_accuracy = 0
                no_improve_count = 0
                
                for epoch in range(EPOCHS):
                    train_epoch(model, train_dl, loss_fn, optimizer, scheduler, scaler, ACCUMULATION_STEPS)
                    true_labels, predictions, _, _ = eval_model(model, valid_dl, loss_fn)
                    accuracy = accuracy_score(true_labels, predictions)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                    if no_improve_count >= EARLY_STOPPING_PATIENCE:
                        logger.info(f"    Early stopping at epoch {epoch + 1}")
                        break

                cv_scores.append(best_accuracy)
                del model, optimizer, scheduler, scaler, train_dl, valid_dl, class_weights
                torch.cuda.empty_cache()
                
            finally:
                if os.path.exists(train_fold_csv):
                    os.remove(train_fold_csv)
                if os.path.exists(valid_fold_csv):
                    os.remove(valid_fold_csv)

        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        logger.info(f"  CV Score: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
        return mean_cv_score

    else:
        train_dl, valid_dl, _, class_weights = prepare_dataloaders(
            train_csv, valid_csv, MODEL_NAME, BATCH_SIZE, MAX_LENGTH, 
            augment_prob=AUGMENT_PROB
        )

        class_weights = class_weights.to(DEVICE)
        loss_fn = CrossEntropyLoss(weight=class_weights)

        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES).to(DEVICE)
        model.gradient_checkpointing_enable()
        
        if hasattr(model.config, 'hidden_dropout_prob'):
            model.config.hidden_dropout_prob = DROPOUT_RATE
            model.config.attention_probs_dropout_prob = DROPOUT_RATE

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        total_steps = (len(train_dl) // ACCUMULATION_STEPS) * EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        scaler = GradScaler()

        torch.cuda.empty_cache()

        best_accuracy = 0
        no_improve_count = 0
        
        for epoch in range(EPOCHS):
            train_epoch(model, train_dl, loss_fn, optimizer, scheduler, scaler, ACCUMULATION_STEPS)
            true_labels, predictions, _, _ = eval_model(model, valid_dl, loss_fn)
            accuracy = accuracy_score(true_labels, predictions)

            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch + 1}")
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
    ACCUMULATION_STEPS = best_params['accumulation_steps']
    WEIGHT_DECAY = best_params.get('weight_decay', 0.01)
    WARMUP_RATIO = best_params.get('warmup_ratio', 0.1)
    DROPOUT_RATE = best_params.get('dropout_rate', 0.1)
    AUGMENT_PROB = best_params.get('augment_prob', 0.3)
    
    logger.info(f"Final training with: LR={LEARNING_RATE:.2e}, BS={BATCH_SIZE}, "
                f"WD={WEIGHT_DECAY:.3f}, Warmup={WARMUP_RATIO:.3f}, Dropout={DROPOUT_RATE:.3f}")

    train_dl, valid_dl, test_dl, class_weights = prepare_dataloaders(
        train_csv, valid_csv, MODEL_NAME, BATCH_SIZE, MAX_LENGTH, augment_prob=AUGMENT_PROB
    )

    class_weights = class_weights.to(DEVICE)
    loss_fn = CrossEntropyLoss(weight=class_weights)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES).to(DEVICE)
    model.gradient_checkpointing_enable()
    
    if hasattr(model.config, 'hidden_dropout_prob'):
        model.config.hidden_dropout_prob = DROPOUT_RATE
        model.config.attention_probs_dropout_prob = DROPOUT_RATE

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = (len(train_dl) // ACCUMULATION_STEPS) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler()

    torch.cuda.empty_cache()

    best_accuracy = 0
    epochs_no_improve = 0
    training_history = {
        'epochs': [],
        'train_loss': [],
        'valid_loss': [],
        'valid_accuracy': [],
        'learning_rate': []
    }

    start_time = time.time()
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        avg_train_loss = train_epoch(model, train_dl, loss_fn, optimizer, scheduler, scaler, ACCUMULATION_STEPS)
        print(f'Training loss: {avg_train_loss}')

        true_labels, predictions, _, avg_val_loss = eval_model(model, valid_dl, loss_fn)
        accuracy = accuracy_score(true_labels, predictions)
        current_lr = optimizer.param_groups[0]['lr']
        elapsed_time = time.time() - start_time
        
        training_history['epochs'].append(epoch + 1)
        training_history['train_loss'].append(avg_train_loss)
        training_history['valid_loss'].append(avg_val_loss)
        training_history['valid_accuracy'].append(accuracy)
        training_history['learning_rate'].append(current_lr)
        
        print(f'Validation Accuracy: {accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, '
              f'LR: {current_lr:.2e} | Time: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save_pretrained(os.path.join(OUTPUT_DIR, 'fine_tuned_model'))
            test_dl.dataset.tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, 'fine_tuned_model'))
            epochs_no_improve = 0
            print("New best model saved.")
            
            with open(os.path.join(OUTPUT_DIR, 'best_hyperparameters.json'), 'w') as f:
                json.dump(best_params, f, indent=2)
                
        else:
            epochs_no_improve += 1

        if epochs_no_improve == EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
    logger.info(f"Best validation accuracy achieved: {best_accuracy:.4f}")

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
