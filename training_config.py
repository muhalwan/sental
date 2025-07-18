import torch

# Model Configuration
MODEL_NAME = 'ProsusAI/finbert'
NUM_CLASSES = 3
MAX_LENGTH = 256

# Training Configuration
EPOCHS = 15
BATCH_SIZE = 16
ACCUMULATION_STEPS = 4
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DROPOUT_RATE = 0.1

# Data Configuration
AUGMENT_PROB = 0.3
TEST_SIZE = 0.1
RANDOM_STATE = 42

# Optimization Configuration
USE_CROSS_VALIDATION = True
N_FOLDS = 5
N_TRIALS = 25
EARLY_STOPPING_PATIENCE = 5

# Output Configuration
OUTPUT_DIR = 'model_outputs'
SAVE_TRAINING_HISTORY = True
SAVE_HYPERPARAMETERS = True

# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_MIXED_PRECISION = True
ENABLE_GRADIENT_CHECKPOINTING = True

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_EVERY_N_STEPS = 100

# Optuna Configuration
OPTUNA_DIRECTION = "maximize"
OPTUNA_PRUNER = "MedianPruner"
OPTUNA_SAMPLER = "TPESampler"

# Hyperparameter Search Spaces
HYPERPARAMETER_SPACES = {
    'lr': {'type': 'float', 'low': 1e-6, 'high': 1e-4, 'log': True},
    'batch_size': {'type': 'categorical', 'choices': [8, 16, 32]},
    'accumulation_steps': {'type': 'categorical', 'choices': [2, 4, 8]},
    'weight_decay': {'type': 'float', 'low': 0.0, 'high': 0.1},
    'warmup_ratio': {'type': 'float', 'low': 0.05, 'high': 0.2},
    'dropout_rate': {'type': 'float', 'low': 0.1, 'high': 0.5},
    'augment_prob': {'type': 'float', 'low': 0.0, 'high': 0.5}
}

# Model Architecture Settings
FREEZE_BERT_LAYERS = False
FREEZE_N_LAYERS = 0

# Advanced Training Settings
GRADIENT_CLIPPING = 1.0
LABEL_SMOOTHING = 0.0
USE_WEIGHTED_LOSS = True
SCHEDULER_TYPE = 'linear' #'cosine', 'polynomial'

# Evaluation Settings
EVAL_EVERY_N_EPOCHS = 1
SAVE_BEST_MODEL = True
EVAL_METRICS = ['accuracy', 'f1_macro', 'f1_weighted', 'precision', 'recall']

# Data Augmentation Settings
AUGMENTATION_TECHNIQUES = ['synonym', 'deletion', 'swap']
AUGMENTATION_PROB_PER_TECHNIQUE = 0.3

# Reproducibility Settings
SEED = 42
DETERMINISTIC = True
