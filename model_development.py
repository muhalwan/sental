import os
import gc
import warnings
import time
import json
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, cast


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

warnings.filterwarnings("ignore", message="expandable_segments not supported on this platform")
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from data_preprocessing import prepare_dataloaders
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import torch.nn.functional as F
import optuna
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_NAME = 'yiyanghkust/finbert-tone'
NUM_CLASSES = 3
EPOCHS = 15
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
MAX_LENGTH = 128
LEARNING_RATE = 3e-5
OUTPUT_DIR = Path('model_outputs')
EARLY_STOPPING_PATIENCE = 5
N_TRIALS = 20
USE_CROSS_VALIDATION = True
N_FOLDS = 5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    DEVICE = torch.device('cpu')
    logger.warning("GPU not available, using CPU. Training will be slower.")


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def train_epoch(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: GradScaler,
        accumulation_steps: int,
        clip_grad_norm: float = 1.0,
        epoch: Optional[int] = None
) -> float:
    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()

    desc = "Training" if epoch is None else f"Training Epoch {epoch + 1}/{EPOCHS}"
    progress_bar = tqdm(data_loader, desc=desc, leave=False)
    for i, batch in enumerate(progress_bar):
        batch = cast(Dict[str, torch.Tensor], batch)
        try:
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            labels = batch['labels'].to(DEVICE, non_blocking=True)

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)
                loss = loss / accumulation_steps

            total_loss += loss.item() * accumulation_steps
            num_batches += 1

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(data_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                current_loss = loss.item() * accumulation_steps
                progress_bar.set_postfix({'loss': f'{current_loss:.4f}',
                                          'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("OOM detected, clearing cache and skipping batch")
                clear_gpu_memory()
                optimizer.zero_grad()
                continue
            else:
                raise e

    return total_loss / max(num_batches, 1)


def eval_model(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        epoch: Optional[int] = None
) -> Tuple[List[int], List[int], np.ndarray, float]:
    model.eval()
    total_loss = 0
    predictions, true_labels, all_probs = [], [], []
    num_batches = 0

    desc = "Evaluating" if epoch is None else f"Evaluating Epoch {epoch + 1}/{EPOCHS}"
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc, leave=False):
            batch = cast(Dict[str, torch.Tensor], batch)
            try:
                input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
                attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
                labels = batch['labels'].to(DEVICE, non_blocking=True)

                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = loss_fn(logits, labels)

                total_loss += loss.item()
                num_batches += 1

                probs = F.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                predictions.extend(preds.tolist())
                true_labels.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.tolist())

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("OOM during evaluation, clearing cache")
                    clear_gpu_memory()
                    continue
                else:
                    raise e

    avg_loss = total_loss / max(num_batches, 1)
    return true_labels, predictions, np.array(all_probs), avg_loss


def objective(trial: optuna.Trial, train_csv: str, valid_csv: str) -> float:
    hp_config = {
        'lr': trial.suggest_float("lr", 1e-6, 1e-4, log=True),
        'batch_size': trial.suggest_categorical("batch_size", [4, 8, 16]),
        'accumulation_steps': trial.suggest_categorical("accumulation_steps", [2, 4, 8]),
        'weight_decay': trial.suggest_float("weight_decay", 0.0, 0.1),
        'warmup_ratio': trial.suggest_float("warmup_ratio", 0.05, 0.2),
        'dropout_rate': trial.suggest_float("dropout_rate", 0.1, 0.5),
        'augment_prob': trial.suggest_float("augment_prob", 0.0, 0.5),
        'gamma': trial.suggest_float("gamma", 1.0, 3.0),
        'eps': trial.suggest_float("eps", 1e-9, 1e-6, log=True),
        'lora_r': trial.suggest_categorical("lora_r", [4, 8, 16]),
        'clip_grad_norm': trial.suggest_float("clip_grad_norm", 0.5, 2.0)
    }

    logger.info(f"Trial {trial.number}: {hp_config}")

    if USE_CROSS_VALIDATION:
        df_train = pd.read_csv(train_csv)
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, df_train['label'])):
            logger.info(f"  Fold {fold + 1}/{N_FOLDS}")

            temp_dir = Path('temp')
            temp_dir.mkdir(exist_ok=True)

            train_fold_csv = temp_dir / f'train_fold_{trial.number}_{fold}.csv'
            valid_fold_csv = temp_dir / f'valid_fold_{trial.number}_{fold}.csv'

            df_train.iloc[train_idx].to_csv(train_fold_csv, index=False)
            df_train.iloc[val_idx].to_csv(valid_fold_csv, index=False)

            try:
                fold_score = train_fold(
                    str(train_fold_csv),
                    str(valid_fold_csv),
                    hp_config,
                    trial,
                    fold
                )
                cv_scores.append(fold_score)

            finally:
                train_fold_csv.unlink(missing_ok=True)
                valid_fold_csv.unlink(missing_ok=True)
                clear_gpu_memory()

        mean_cv_score = float(np.mean(cv_scores))
        std_cv_score = np.std(cv_scores)
        logger.info(f"  CV Score: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
        return mean_cv_score

    else:
        return train_fold(train_csv, valid_csv, hp_config, trial, fold=0)


def train_fold(
        train_csv: str,
        valid_csv: str,
        hp_config: Dict[str, Any],
        trial: optuna.Trial,
        fold: int
) -> float:
    try:
        train_dl, valid_dl, _, class_weights = prepare_dataloaders(
            train_csv, valid_csv, MODEL_NAME,
            hp_config['batch_size'], MAX_LENGTH,
            augment_prob=hp_config['augment_prob']
        )

        class_weights = class_weights.to(DEVICE)
        loss_fn = FocalLoss(alpha=class_weights, gamma=hp_config['gamma'])

        model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_CLASSES,
            hidden_dropout_prob=hp_config['dropout_rate'],
            attention_probs_dropout_prob=hp_config['dropout_rate']
        )

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=hp_config['lora_r'],
            lora_alpha=32,
            lora_dropout=hp_config['dropout_rate'],
            bias="none",
            target_modules=["query", "value"]
        )
        model = get_peft_model(model, lora_config)
        model = model.to(DEVICE)

        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        optimizer = AdamW(
            model.parameters(),
            lr=hp_config['lr'],
            weight_decay=hp_config['weight_decay'],
            eps=hp_config['eps']
        )

        total_steps = (len(train_dl) // hp_config['accumulation_steps']) * EPOCHS
        warmup_steps = int(total_steps * hp_config['warmup_ratio'])

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        scaler = GradScaler(init_scale=2.**16)

        best_accuracy = 0
        no_improve_count = 0

        for epoch in range(EPOCHS):
            train_loss = train_epoch(
                model, train_dl, loss_fn, optimizer, scheduler, scaler,
                hp_config['accumulation_steps'], hp_config['clip_grad_norm'],
                epoch=epoch
            )

            true_labels, predictions, _, val_loss = eval_model(model, valid_dl, loss_fn, epoch=epoch)
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
                logger.info(f"    Early stopping at epoch {epoch + 1}")
                break

        return best_accuracy

    except Exception as e:
        logger.error(f"Error in fold {fold}: {str(e)}")
        raise

    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if 'optimizer' in locals():
            del optimizer
        if 'scheduler' in locals():
            del scheduler
        if 'scaler' in locals():
            del scaler
        clear_gpu_memory()


def plot_visualizations(
        true_labels: List[int],
        predictions: List[int],
        output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        true_labels, predictions,
        target_names=['Bearish', 'Bullish', 'Neutral'],
        output_dict=True
    )

    with open(output_dir / 'classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("\nClassification Report:")
    print(classification_report(
        true_labels, predictions,
        target_names=['Bearish', 'Bullish', 'Neutral']
    ))

    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Bearish', 'Bullish', 'Neutral'],
        yticklabels=['Bearish', 'Bullish', 'Neutral']
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=100)
    plt.close()


def plot_roc_curves(
        true_labels: List[int],
        predictions_prob: np.ndarray,
        output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    true_labels_bin = label_binarize(true_labels, classes=[0, 1, 2])
    n_classes = true_labels_bin.shape[1]

    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], predictions_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    class_names = ['Bearish', 'Bullish', 'Neutral']

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f'ROC curve of {class_names[i]} (AUC = {roc_auc[i]:.2f})'
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=100)
    plt.close()


def visualize_attention(
        model: torch.nn.Module,
        tokenizer: Any,
        sentences: List[str],
        output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    for i, sentence in enumerate(sentences):
        try:
            inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=MAX_LENGTH)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)

            attentions = outputs.attentions[-1]
            attention = torch.mean(attentions, dim=1).squeeze(0)
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

            plt.figure(figsize=(12, 10))
            sns.heatmap(
                attention.cpu().numpy(),
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='viridis'
            )
            plt.title(f'Attention Heatmap for: "{sentence[:50]}..."')
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(output_dir / f'attention_heatmap_{i + 1}.png', dpi=100, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Error visualizing attention for sentence {i + 1}: {str(e)}")


def save_model_artifacts(
        model: torch.nn.Module,
        tokenizer: Any,
        best_params: Dict[str, Any],
        training_history: Dict[str, List],
        output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir / 'fine_tuned_model')
    tokenizer.save_pretrained(output_dir / 'fine_tuned_model')

    with open(output_dir / 'best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)

    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    model.config.to_json_file(output_dir / 'config.json')

    logger.info(f"Model artifacts saved to {output_dir}")


def export_to_onnx(
        model: torch.nn.Module,
        output_dir: Path,
        max_length: int = MAX_LENGTH
) -> None:
    try:
        onnx_path = output_dir / 'model.onnx'

        dummy_input = torch.zeros(1, max_length, dtype=torch.long).to(DEVICE)
        dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long).to(DEVICE)

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
            export_params=True,
            do_constant_folding=True
        )

        logger.info(f"ONNX model saved to {onnx_path}")

    except Exception as e:
        logger.error(f"Failed to export ONNX model: {str(e)}")


def setup_paths():
    base_dir = Path(__file__).parent
    train_csv = base_dir / 'dataset' / 'sent_train.csv'
    valid_csv = base_dir / 'dataset' / 'sent_valid.csv'

    if not train_csv.exists() or not valid_csv.exists():
        raise FileNotFoundError(f"Dataset files not found in {base_dir / 'dataset'}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    return train_csv, valid_csv


def optimize_hyperparameters(train_csv: str, valid_csv: str):
    from optuna.trial import TrialState

    logger.info("=" * 50)
    logger.info("Starting hyperparameter optimization with Optuna...")
    logger.info("=" * 50)

    storage = f"sqlite:///{OUTPUT_DIR / 'optuna_study.db'}"
    study_name = "financial_sentiment_optimization"

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            sampler=optuna.samplers.TPESampler(seed=SEED),
            load_if_exists=True
        )

    running_trials = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    for trial in running_trials:
        logger.info(f"Failing incomplete running trial {trial.number}")
        study.tell(trial.number, state=TrialState.FAIL)

    completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    pruned_trials = study.get_trials(deepcopy=False, states=(TrialState.PRUNED,))
    failed_trials = study.get_trials(deepcopy=False, states=(TrialState.FAIL,))
    running_trials = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))

    logger.info(
        f"Study status: "
        f"{len(completed_trials)} completed, {len(pruned_trials)} pruned, "
        f"{len(failed_trials)} failed, {len(running_trials)} running."
    )

    completed_count = len(completed_trials)
    remaining_trials = max(N_TRIALS - completed_count, 0)
    last_number = max([t.number for t in study.trials], default=-1)

    if remaining_trials > 0:
        logger.info(
            f"{completed_count} trials completed out of target {N_TRIALS}. "
            f"Scheduling {remaining_trials} more. Next trial number will start at {last_number + 1}."
        )
        study.optimize(
            lambda t: objective(t, train_csv, valid_csv),
            n_trials=remaining_trials,
            timeout=3600 * 6,
            gc_after_trial=True
        )
    else:
        logger.info(
            f"{completed_count} trials completed. Target {N_TRIALS} already reached. Skipping optimization."
        )

    logger.info("=" * 50)
    logger.info("Optimization finished.")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best validation accuracy: {study.best_trial.value:.4f}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        logger.info(f"  {key}: {value}")

    study_df = study.trials_dataframe()
    study_df.to_csv(OUTPUT_DIR / 'optuna_study_results.csv', index=False)

    return study


def train_final_model(study, train_csv: str, valid_csv: str):
    logger.info("=" * 50)
    logger.info("Starting final training with best hyperparameters...")
    logger.info("=" * 50)

    best_params = study.best_trial.params

    train_dl, valid_dl, test_dl, class_weights = prepare_dataloaders(
        train_csv, valid_csv, MODEL_NAME,
        best_params['batch_size'], MAX_LENGTH,
        augment_prob=best_params.get('augment_prob', 0.3)
    )

    class_weights = class_weights.to(DEVICE)
    loss_fn = FocalLoss(alpha=class_weights, gamma=best_params.get('gamma', 2.0))

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        hidden_dropout_prob=best_params.get('dropout_rate', 0.1),
        attention_probs_dropout_prob=best_params.get('dropout_rate', 0.1)
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=best_params.get('lora_r', 8),
        lora_alpha=32,
        lora_dropout=best_params.get('dropout_rate', 0.1),
    )
    model = get_peft_model(model, lora_config)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params['lr'],
        weight_decay=best_params.get('weight_decay', 0.01),
        eps=best_params.get('eps', 1e-8)
    )

    total_steps = len(train_dl) * EPOCHS // best_params.get('accumulation_steps', 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * best_params.get('warmup_ratio', 0.1)),
        num_training_steps=total_steps
    )

    best_val_acc = -1.0
    epochs_no_improve = 0
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        train_loss = train_epoch(
            model, train_dl, loss_fn, optimizer, scheduler, scaler,
            accumulation_steps=best_params.get('accumulation_steps', 1),
            clip_grad_norm=best_params.get('clip_grad_norm', 1.0),
            epoch=epoch-1
        )
        true_labels, predictions, _, val_loss = eval_model(model, valid_dl, loss_fn, epoch=epoch-1)
        val_acc = accuracy_score(true_labels, predictions)

        elapsed = time.time() - start_time
        logger.info(
            f"\nEpoch {epoch}/{EPOCHS}\n"
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {int(elapsed//60)}m {int(elapsed%60)}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_model_artifacts(
                model=model,
                tokenizer=tokenizer,
                best_params=best_params,
                training_history={},
                output_dir=OUTPUT_DIR
            )
            logger.info("New best model saved.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    export_to_onnx(model, OUTPUT_DIR)

    return model, test_dl, loss_fn, tokenizer


def perform_evaluation_and_visualizations(model, test_dl, loss_fn, tokenizer, output_dir: Path):
    true_labels, predictions, probs, _ = eval_model(model, test_dl, loss_fn)

    plot_visualizations(true_labels, predictions, output_dir)
    plot_roc_curves(true_labels, probs, output_dir)

    logger.info("\nGenerating attention visualizations for sample texts...")
    sample_texts = [
        "BTIG points to breakfast pressure for Dunkin' Brands",
        "$CX - Cemex cut at Credit Suisse, J.P. Morgan on weak building outlook",
        "Adobe price target raised to $350 vs. $320 at Canaccord"
    ]
    visualize_attention(model, tokenizer, sample_texts, output_dir)

    logger.info(f"\nModel and visualizations saved in '{output_dir}' directory.")


def main():
    train_csv, valid_csv = setup_paths()

    study = optimize_hyperparameters(str(train_csv), str(valid_csv))

    model, test_dl, loss_fn, tokenizer = train_final_model(study, str(train_csv), str(valid_csv))

    perform_evaluation_and_visualizations(model, test_dl, loss_fn, tokenizer, OUTPUT_DIR)


if __name__ == '__main__':
    main()