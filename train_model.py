import os
import sys
import json
import logging
import numpy as np
import pandas as pd

os.environ['USE_TF'] = 'NO'
os.environ['USE_TORCH'] = 'YES'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import optuna

from data_preprocessing import prepare_dataloaders
from model_development import train_epoch, eval_model, plot_visualizations, plot_roc_curves
from training_config import *

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{OUTPUT_DIR}/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    if DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def calculate_metrics(true_labels, predictions):
    """Calculate comprehensive evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'f1_macro': f1_score(true_labels, predictions, average='macro'),
        'f1_weighted': f1_score(true_labels, predictions, average='weighted'),
        'precision': precision_score(true_labels, predictions, average='weighted'),
        'recall': recall_score(true_labels, predictions, average='weighted')
    }
    return metrics

def create_model_with_config(dropout_rate=DROPOUT_RATE):
    """Create model with specified configuration."""
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES).to(DEVICE)
    
    if ENABLE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    
    if hasattr(model.config, 'hidden_dropout_prob'):
        model.config.hidden_dropout_prob = dropout_rate
        model.config.attention_probs_dropout_prob = dropout_rate
    
    if FREEZE_BERT_LAYERS:
        for param in model.bert.parameters():
            param.requires_grad = False
    elif FREEZE_N_LAYERS > 0:
        for i in range(FREEZE_N_LAYERS):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = False
    
    return model

def objective(trial, train_csv, valid_csv):
    """Optuna objective function with improved hyperparameter search."""
    params = {}
    for param_name, param_config in HYPERPARAMETER_SPACES.items():
        if param_config['type'] == 'float':
            params[param_name] = trial.suggest_float(
                param_name, param_config['low'], param_config['high'], 
                log=param_config.get('log', False)
            )
        elif param_config['type'] == 'categorical':
            params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
        elif param_config['type'] == 'int':
            params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
    
    logger.info(f"Trial {trial.number}: {params}")
    
    try:
        if USE_CROSS_VALIDATION:
            return cross_validate_trial(params, train_csv, valid_csv, trial)
        else:
            return single_validation_trial(params, train_csv, valid_csv, trial)
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {str(e)}")
        raise optuna.exceptions.TrialPruned()

def single_validation_trial(params, train_csv, valid_csv, trial):
    """Perform single validation for a trial."""
    try:
        train_dl, valid_dl, _, class_weights = prepare_dataloaders(
            train_csv, valid_csv, MODEL_NAME, 
            params['batch_size'], MAX_LENGTH, augment_prob=params['augment_prob']
        )
        
        score = train_fold(train_dl, valid_dl, class_weights, params, trial, 0)
        
        del train_dl, valid_dl, class_weights
        torch.cuda.empty_cache()
        
        return score
        
    except Exception as e:
        logger.error(f"Single validation trial failed: {str(e)}")
        raise optuna.exceptions.TrialPruned()

def cross_validate_trial(params, train_csv, valid_csv, trial):
    """Perform cross-validation for a single trial."""
    df_train = pd.read_csv(train_csv)
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, df_train['label'])):
        logger.info(f"  Fold {fold + 1}/{N_FOLDS}")
        
        train_fold_df = df_train.iloc[train_idx].reset_index(drop=True)
        valid_fold_df = df_train.iloc[val_idx].reset_index(drop=True)
        
        train_fold_csv = f'temp_train_fold_{fold}.csv'
        valid_fold_csv = f'temp_valid_fold_{fold}.csv'
        
        try:
            train_fold_df.to_csv(train_fold_csv, index=False)
            valid_fold_df.to_csv(valid_fold_csv, index=False)
            
            train_dl, valid_dl, _, class_weights = prepare_dataloaders(
                train_fold_csv, valid_fold_csv, MODEL_NAME, 
                params['batch_size'], MAX_LENGTH, augment_prob=params['augment_prob']
            )
            
            fold_score = train_fold(
                train_dl, valid_dl, class_weights, params, trial, fold
            )
            cv_scores.append(fold_score)
            
            del train_dl, valid_dl, class_weights
            torch.cuda.empty_cache()
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    logger.info(f"  CV Score: {mean_score:.4f} ± {std_score:.4f}")
    
    return mean_score

def train_fold(train_dl, valid_dl, class_weights, params, trial, fold):
    """Train model for a single fold."""
    class_weights = class_weights.to(DEVICE)
    loss_fn = CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    
    model = create_model_with_config(params['dropout_rate'])
    
    optimizer = AdamW(
        model.parameters(), 
        lr=params['lr'], 
        weight_decay=params['weight_decay']
    )
    
    total_steps = (len(train_dl) // params['accumulation_steps']) * EPOCHS
    warmup_steps = int(total_steps * params['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    scaler = GradScaler() if USE_MIXED_PRECISION else None
    
    best_score = 0
    no_improve_count = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_dl):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            if USE_MIXED_PRECISION:
                with autocast(device_type='cuda'):
                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits, labels) / params['accumulation_steps']
                
                scaler.scale(loss).backward()
                
                if (i + 1) % params['accumulation_steps'] == 0:
                    if GRADIENT_CLIPPING > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels) / params['accumulation_steps']
                loss.backward()
                
                if (i + 1) % params['accumulation_steps'] == 0:
                    if GRADIENT_CLIPPING > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item()
        
        true_labels, predictions, _, val_loss = eval_model(model, valid_dl, loss_fn)
        metrics = calculate_metrics(true_labels, predictions)
        
        current_score = metrics['accuracy']
        
        if current_score > best_score:
            best_score = current_score
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= EARLY_STOPPING_PATIENCE:
            logger.info(f"    Early stopping at epoch {epoch + 1}")
            break
        
        if fold == 0:
            trial.report(current_score, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
    del model, optimizer, scheduler, loss_fn
    if scaler:
        del scaler
    torch.cuda.empty_cache()
    
    return best_score

def final_training_with_best_params(best_params, train_csv, valid_csv):
    """Run final training with the best hyperparameters."""
    import time
    
    logger.info(f"Final training with best parameters: {best_params}")
    
    train_dl, valid_dl, test_dl, class_weights = prepare_dataloaders(
        train_csv, valid_csv, MODEL_NAME, 
        best_params['batch_size'], MAX_LENGTH, 
        augment_prob=best_params['augment_prob']
    )
    
    class_weights = class_weights.to(DEVICE)
    loss_fn = CrossEntropyLoss(weight=class_weights)
    
    model = create_model_with_config(best_params['dropout_rate'])
    
    optimizer = AdamW(
        model.parameters(), 
        lr=best_params['lr'], 
        weight_decay=best_params['weight_decay']
    )
    
    total_steps = (len(train_dl) // best_params['accumulation_steps']) * EPOCHS
    warmup_steps = int(total_steps * best_params['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    scaler = GradScaler()
    
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
        logger.info(f'Epoch {epoch + 1}/{EPOCHS}')
        
        avg_train_loss = train_epoch(
            model, train_dl, loss_fn, optimizer, scheduler, scaler, 
            best_params['accumulation_steps']
        )
        
        true_labels, predictions, _, avg_val_loss = eval_model(model, valid_dl, loss_fn)
        accuracy = accuracy_score(true_labels, predictions)
        current_lr = optimizer.param_groups[0]['lr']
        
        training_history['epochs'].append(epoch + 1)
        training_history['train_loss'].append(avg_train_loss)
        training_history['valid_loss'].append(avg_val_loss)
        training_history['valid_accuracy'].append(accuracy)
        training_history['learning_rate'].append(current_lr)
        
        elapsed_time = time.time() - start_time
        logger.info(f'Train Loss: {avg_train_loss:.4f}, Valid Acc: {accuracy:.4f}, '
                   f'Valid Loss: {avg_val_loss:.4f}, LR: {current_lr:.2e}, '
                   f'Time: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_no_improve = 0
            
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(os.path.join(OUTPUT_DIR, 'fine_tuned_model'))
            test_dl.dataset.tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, 'fine_tuned_model'))
            
            with open(os.path.join(OUTPUT_DIR, 'best_hyperparameters.json'), 'w') as f:
                json.dump(best_params, f, indent=2)
            
            logger.info("New best model saved!")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
            break
    
    with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("Loading best model for final evaluation...")
    model = BertForSequenceClassification.from_pretrained(
        os.path.join(OUTPUT_DIR, 'fine_tuned_model')
    ).to(DEVICE)
    
    true_labels, predictions, probs, _ = eval_model(model, test_dl, loss_fn)
    
    plot_visualizations(true_labels, predictions, OUTPUT_DIR)
    plot_roc_curves(true_labels, probs, OUTPUT_DIR)
    
    logger.info(f"Final training completed! Best accuracy: {best_accuracy:.4f}")
    logger.info(f"Results saved in '{OUTPUT_DIR}' directory.")

def main():
    """Main training function."""
    set_seed(SEED)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"Device: {DEVICE}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    
    base_dir = os.path.dirname(__file__)
    train_csv = os.path.join(base_dir, 'dataset', 'sent_train.csv')
    valid_csv = os.path.join(base_dir, 'dataset', 'sent_valid.csv')
    
    if not os.path.exists(train_csv) or not os.path.exists(valid_csv):
        logger.error("Training or validation data files not found!")
        return
    
    config_dict = {k: v for k, v in globals().items()
                   if not k.startswith('_') and isinstance(v, (int, float, str, bool, list, dict))}
    
    with open(os.path.join(OUTPUT_DIR, 'training_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    logger.info("Starting hyperparameter optimization...")
    
    study = optuna.create_study(
        direction=OPTUNA_DIRECTION,
        pruner=getattr(optuna.pruners, OPTUNA_PRUNER)() if OPTUNA_PRUNER else None,
        sampler=getattr(optuna.samplers, OPTUNA_SAMPLER)() if OPTUNA_SAMPLER else None
    )
    
    study.optimize(
        lambda trial: objective(trial, train_csv, valid_csv),
        n_trials=N_TRIALS
    )
    
    logger.info("Optimization completed!")
    logger.info(f"Best value: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    study_results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'study_name': study.study_name
    }
    
    with open(os.path.join(OUTPUT_DIR, 'optimization_results.json'), 'w') as f:
        json.dump(study_results, f, indent=2)
    
    logger.info("Starting final training with best parameters...")
    
    final_training_with_best_params(study.best_params, train_csv, valid_csv)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
