import os
import gc
import warnings
import time
import json
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import math

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

warnings.filterwarnings("ignore", message="expandable_segments not supported on this platform")
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from data_preprocessing import prepare_dataloaders
from tqdm import tqdm
import torch.nn.functional as F
import optuna
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import ml_collections
from contextlib import nullcontext
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_NAME = 'yiyanghkust/finbert-tone'
NUM_CLASSES = 3
EPOCHS = 15
BATCH_SIZE = 16
ACCUMULATION_STEPS = 4
MAX_LENGTH = 128
LEARNING_RATE = 3e-5
OUTPUT_DIR = Path('model_outputs')
EARLY_STOPPING_PATIENCE = 3
N_TRIALS = 20
USE_CROSS_VALIDATION = True
N_FOLDS = 5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _space_signature():
    space = {
        "lr": ("float_log", 1e-6, 1e-4),
        "batch_size": ("cat", [8, 16, 32]),
        "accumulation_steps": ("cat", [2, 4, 8]),
        "weight_decay": ("float", 0.0, 0.1),
        "warmup_ratio": ("float", 0.05, 0.2),
        "dropout_rate": ("float", 0.1, 0.5),
        "augment_prob": ("float", 0.0, 0.5),
        "gamma": ("float", 1.0, 3.0),
        "eps": ("float_log", 1e-9, 1e-6),
        "lora_r": ("cat", [4, 8, 16]),
        "clip_grad_norm": ("float", 0.5, 2.0),
    }
    blob = json.dumps(space, sort_keys=True).encode()
    return hashlib.md5(blob).hexdigest()[:8]


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if __name__ == '__main__':
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        if __name__ == '__main__':
            logger.warning("GPU not available, using CPU. Training will be slower.")
    return device


DEVICE = get_device()


def move_to_device(batch, device):
    import numpy as _np
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, _np.ndarray):
        return torch.from_numpy(batch).to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device) for v in batch)
    return batch


def setup_nltk():
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
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
        torch.cuda.synchronize()
    gc.collect()


def model_config():
    cfg_dictionary = {
        "data_path": "dataset/sent_train.csv",
        "model_path": "/models/bert_model.h5",
        "model_type": "transformer",
        "test_size": 0.1,
        "validation_size": 0.2,
        "train_batch_size": 32,
        "eval_batch_size": 32,
        "epochs": 5,
        "adam_epsilon": 1e-8,
        "lr": 3e-5,
        "num_warmup_steps": 10,
        "max_length": 128,
        "random_seed": 42,
        "num_labels": 3,
        "model_checkpoint": 'yiyanghkust/finbert-tone',
    }
    cfg = ml_collections.FrozenConfigDict(cfg_dictionary)
    return cfg


def preprocess_csv(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    df["label_enc"] = labelencoder.fit_transform(df["Sentiment"])
    df.rename(columns={"label": "label_desc"}, inplace=True)
    df.rename(columns={"label_enc": "labels"}, inplace=True)
    df.drop_duplicates(subset=['Sentence'], keep='first', inplace=True)
    return df


def perform_eda(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        from pandas_profiling import ProfileReport

    profile = ProfileReport(df, title="Financial Sentiment Analysis", minimal=True)
    profile.to_file(output_dir / "eda_report.html")


def ml_based_approach(df: pd.DataFrame):
    setup_nltk()
    from nltk.tokenize import word_tokenize

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(df["Sentence"]), np.array(df["labels"]), test_size=0.25, random_state=SEED
    )

    tfidf = TfidfVectorizer(use_idf=True, tokenizer=word_tokenize, min_df=0.00002, max_df=0.70)
    x_train_tf = tfidf.fit_transform(x_train.astype('U'))
    x_test_tf = tfidf.transform(x_test.astype('U'))

    clfs = {
        "Random Forest": RandomForestClassifier(random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "AdaBoost": AdaBoostClassifier(random_state=SEED),
        "LightGBM": LGBMClassifier(random_state=SEED, verbose=-1),
        "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=SEED, verbosity=0),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Support Vector Machine": SVC(random_state=SEED),
        "Naive Bayes": MultinomialNB(),
        "Multilayer Perceptron": MLPClassifier(random_state=SEED, max_iter=500)
    }

    accuracies = []
    for name, clf in tqdm(clfs.items(), desc="Training ML Models"):
        clf.fit(x_train_tf, y_train)
        y_pred = clf.predict(x_test_tf)
        accuracy = accuracy_score(y_pred, y_test)
        accuracies.append(accuracy)

    models_df = pd.DataFrame({"Models": clfs.keys(), "Accuracy Scores": accuracies}).sort_values(
        'Accuracy Scores', ascending=False
    )
    return models_df


def train_epoch(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loss_fn: torch.nn.Module,
        accumulation_steps: int,
        clip_grad_norm: float = 1.0,
        epoch: Optional[int] = None,
        accelerator: Accelerator = None
) -> float:
    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()

    desc = "Training" if epoch is None else f"Training Epoch {epoch + 1}/{EPOCHS}"
    progress_bar = tqdm(data_loader, desc=desc, leave=False)

    for i, batch in enumerate(progress_bar):
        target_device = accelerator.device if accelerator is not None else DEVICE
        batch = move_to_device(batch, target_device)

        try:
            ctx = accelerator.autocast() if accelerator is not None else nullcontext()
            with ctx:
                outputs = model(**batch)
                logits = outputs.logits
                labels = batch["labels"]
                loss = loss_fn(logits, labels) / accumulation_steps

            if accelerator is not None:
                accelerator.backward(loss)
            else:
                loss.backward()

            total_loss += loss.item() * accumulation_steps
            num_batches += 1

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(data_loader):
                if accelerator is not None:
                    accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

        except RuntimeError as exc:
            if "out of memory" in str(exc):
                logger.warning("OOM detected, clearing cache and skipping batch")
                clear_gpu_memory()
                optimizer.zero_grad()
                continue
            else:
                raise exc

    return total_loss / max(num_batches, 1)


def eval_model(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        epoch: Optional[int] = None,
        accelerator: Accelerator = None
) -> Tuple[List[int], List[int], np.ndarray, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    predictions: List[int] = []
    true_labels: List[int] = []
    all_probs_list: List[List[float]] = []

    desc = "Evaluating" if epoch is None else f"Evaluating Epoch {epoch + 1}/{EPOCHS}"

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc, leave=False):
            target_device = accelerator.device if accelerator is not None else DEVICE
            batch = move_to_device(batch, target_device)

            ctx = accelerator.autocast() if accelerator is not None else nullcontext()
            with ctx:
                outputs = model(**batch)
                logits = outputs.logits
                labels = batch["labels"]
                loss = loss_fn(logits, labels)

            total_loss += loss.item()
            num_batches += 1

            probs_t = F.softmax(logits, dim=1)
            preds_t = torch.argmax(probs_t, dim=1)

            if accelerator is not None:
                preds_g = accelerator.gather_for_metrics(preds_t)
                labels_g = accelerator.gather_for_metrics(labels)
                probs_g = accelerator.gather_for_metrics(probs_t)
            else:
                preds_g, labels_g, probs_g = preds_t, labels, probs_t

            predictions.extend(preds_g.detach().cpu().tolist())
            true_labels.extend(labels_g.detach().cpu().tolist())
            all_probs_list.extend(probs_g.detach().cpu().tolist())

    avg_loss = total_loss / max(num_batches, 1)
    return true_labels, predictions, np.asarray(all_probs_list), avg_loss


def objective(trial: optuna.Trial, train_csv: str, valid_csv: str) -> float:
    hp_config = {
        'lr': trial.suggest_float("lr", 1e-6, 1e-4, log=True),
        'batch_size': trial.suggest_categorical("batch_size", [8, 16, 32]),
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

    if USE_CROSS_VALIDATION:
        df_train = pd.read_csv(train_csv)
        from sklearn.preprocessing import LabelEncoder
        labelencoder = LabelEncoder()
        df_train["label"] = labelencoder.fit_transform(df_train["Sentiment"])

        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, df_train['label'])):
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
        accelerator = Accelerator(mixed_precision='fp16', log_with=None)

        train_dl, valid_dl, _, class_weights = prepare_dataloaders(
            train_csv, valid_csv, MODEL_NAME,
            hp_config['batch_size'], MAX_LENGTH,
            augment_prob=hp_config['augment_prob']
        )

        class_weights = class_weights.to(accelerator.device)
        loss_fn = FocalLoss(alpha=class_weights, gamma=hp_config['gamma'])

        model = AutoModelForSequenceClassification.from_pretrained(
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

        optimizer = AdamW(
            model.parameters(),
            lr=hp_config['lr'],
            weight_decay=hp_config['weight_decay'],
            eps=hp_config['eps']
        )

        total_steps = math.ceil(len(train_dl) / hp_config['accumulation_steps']) * EPOCHS
        warmup_steps = max(1, int(total_steps * hp_config['warmup_ratio']))

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        model, optimizer, train_dl, valid_dl, scheduler, loss_fn = accelerator.prepare(
            model, optimizer, train_dl, valid_dl, scheduler, loss_fn
        )

        best_accuracy = 0
        no_improve_count = 0

        for epoch in range(EPOCHS):
            train_loss = train_epoch(
                model, train_dl, optimizer, scheduler, loss_fn,
                hp_config['accumulation_steps'], hp_config['clip_grad_norm'],
                epoch=epoch, accelerator=accelerator
            )

            true_labels, predictions, _, val_loss = eval_model(
                model, valid_dl, loss_fn, epoch=epoch, accelerator=accelerator
            )

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
                break

        return best_accuracy

    except Exception as exc:
        logger.error(f"Error in fold {fold}: {str(exc)}")
        raise

    finally:
        locals_to_del = ['model', 'optimizer', 'scheduler', 'accelerator']
        for var_name in locals_to_del:
            if var_name in locals():
                del locals()[var_name]
        clear_gpu_memory()


def plot_visualizations(
        true_labels: List[int],
        predictions: List[int],
        output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        true_labels, predictions,
        target_names = ['Negative', 'Neutral', 'Positive'],
        output_dict = True
    )

    with open(output_dir / 'classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("\nClassification Report:")
    print(classification_report(
        true_labels, predictions,
        target_names = ['Negative', 'Neutral', 'Positive']
    ))

    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Negative', 'Neutral', 'Positive'],
        yticklabels=['Negative', 'Neutral', 'Positive']
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
    class_names = ['Negative', 'Neutral', 'Positive']

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

        except Exception as exc:
            logger.error(f"Error visualizing attention for sentence {i + 1}: {str(exc)}")


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
        model.eval()
        orig_return_dict = getattr(model.config, "return_dict", True)
        model.config.return_dict = False

        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_input, dummy_attention_mask),
                onnx_path,
                input_names = ['input_ids', 'attention_mask'],
                output_names = ['logits'],
                dynamic_axes = {
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence'},
                    'logits': {0: 'batch_size'}
                },
                opset_version = 14,
                export_params = True,
                do_constant_folding = True
            )
        model.config.return_dict = orig_return_dict
        logger.info(f"ONNX model saved to {onnx_path}")

    except Exception as exc:
        logger.error(f"Failed to export ONNX model: {str(exc)}")


def plot_comparison(models_df: pd.DataFrame, transformer_acc: float, output_dir: Path):
    transformer_row = pd.DataFrame([{'Models': 'FinBERT', 'Accuracy Scores': transformer_acc}])
    models_df = pd.concat([models_df, transformer_row], ignore_index=True)
    models_df = models_df.sort_values('Accuracy Scores', ascending=False)

    plt.rcParams['figure.figsize'] = 22, 10
    sns.set_style("darkgrid")
    ax = sns.barplot(x=models_df["Models"], y=models_df["Accuracy Scores"], palette="coolwarm", saturation=1.5)
    plt.xlabel("Classification Models", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.title("Accuracy of different Classification Models", fontsize=20)
    plt.xticks(fontsize=11, horizontalalignment='center', rotation=8)
    plt.yticks(fontsize=13)

    for p in ax.patches:
        if isinstance(p, patches.Rectangle):
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            ax.annotate(f'{height:.2%}', (x + width / 2, y + height * 1.02), ha='center', fontsize='x-large')

    plt.savefig(output_dir / 'model_comparison.png', dpi=100)
    plt.close()


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

    sig = _space_signature()
    study_name = f"financial_sentiment_optimization__{sig}"
    storage = f"sqlite:///{OUTPUT_DIR / f'optuna_study__{sig}.db'}"

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            sampler=optuna.samplers.TPESampler(seed=SEED),
            load_if_exists=True
        )

    running = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    for t in running:
        study.tell(t.number, state=TrialState.FAIL)

    completed = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    remaining = max(N_TRIALS - len(completed), 0)

    if remaining > 0:
        study.optimize(lambda trl: objective(trl, train_csv, valid_csv),
                       n_trials=remaining, timeout=3600 * 6, gc_after_trial=True)

    logger.info("=" * 50)
    logger.info("Optimization finished.")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best validation accuracy: {study.best_trial.value:.4f}")

    study_df = study.trials_dataframe()
    study_df.to_csv(OUTPUT_DIR / f'optuna_study_results__{sig}.csv', index=False)
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

    accelerator = Accelerator(mixed_precision='fp16', log_with=None)

    class_weights = class_weights.to(accelerator.device)
    loss_fn = FocalLoss(alpha=class_weights, gamma=best_params.get('gamma', 2.0))

    model = AutoModelForSequenceClassification.from_pretrained(
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
        target_modules=["query", "value"]
    )

    model = get_peft_model(model, lora_config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params['lr'],
        weight_decay=best_params.get('weight_decay', 0.01),
        eps=best_params.get('eps', 1e-8)
    )

    total_steps = math.ceil(len(train_dl) / best_params.get('accumulation_steps', 1)) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * best_params.get('warmup_ratio', 0.1)),
        num_training_steps=total_steps
    )

    model, optimizer, train_dl, valid_dl, scheduler, loss_fn = accelerator.prepare(
        model, optimizer, train_dl, valid_dl, scheduler, loss_fn
    )

    best_val_acc = -1.0
    epochs_no_improve = 0
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_dl, optimizer, scheduler, loss_fn,
            accumulation_steps=best_params.get('accumulation_steps', 1),
            clip_grad_norm=best_params.get('clip_grad_norm', 1.0),
            epoch=epoch - 1,
            accelerator=accelerator
        )

        true_labels, predictions, _, val_loss = eval_model(
            model, valid_dl, loss_fn, epoch=epoch - 1, accelerator=accelerator
        )

        val_acc = accuracy_score(true_labels, predictions)
        elapsed = time.time() - start_time

        logger.info(
            f"\nEpoch {epoch}/{EPOCHS}\n"
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {int(elapsed // 60)}m {int(elapsed % 60)}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            unwrapped_model = accelerator.unwrap_model(model)
            save_model_artifacts(
                model=unwrapped_model,
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

    unwrapped_model = accelerator.unwrap_model(model)
    export_to_onnx(unwrapped_model, OUTPUT_DIR)

    return unwrapped_model, test_dl, loss_fn, tokenizer


def perform_evaluation_and_visualizations(model, test_dl, loss_fn, tokenizer, output_dir: Path,
                                          models_df: pd.DataFrame):
    true_labels, predictions, probs, _ = eval_model(model, test_dl, loss_fn)
    transformer_acc = accuracy_score(true_labels, predictions)

    plot_comparison(models_df, transformer_acc, output_dir)
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
    cfg = model_config()
    train_csv, valid_csv = setup_paths()

    df = preprocess_csv(cfg.data_path)
    perform_eda(df, OUTPUT_DIR)

    models_df = ml_based_approach(df)

    study = optimize_hyperparameters(str(train_csv), str(valid_csv))

    model, test_dl, loss_fn, tokenizer = train_final_model(study, str(train_csv), str(valid_csv))

    perform_evaluation_and_visualizations(model, test_dl, loss_fn, tokenizer, OUTPUT_DIR, models_df)


if __name__ == '__main__':
    import multiprocessing as mp

    mp.set_start_method('spawn', force=True)

    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
