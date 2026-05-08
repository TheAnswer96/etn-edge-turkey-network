"""
Global configuration for turkey detection model.
All parameters are hardcoded here for easy modification.
"""

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
IMAGE_DIR = 'src/dataset/images'
XML_DIR = 'src/dataset/xml_labels'
IMG_WIDTH = 416
IMG_HEIGHT = 312
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
HEATMAP_STRIDE = 32  # Total stride from model (5 downsampling layers: 2^5=32)
GAUSSIAN_SIGMA = 2.0
TEST_SPLIT = 0.15
AUGMENT = True
NUM_WORKERS = 0  # Set to 0 for Windows, increase on Linux for speed
PIN_MEMORY = True
SEED = 42

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
BACKBONE_CHANNELS = 1024
HEATMAP_CHANNELS = 1
NUM_CLASSES = 2

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_MAX_NORM = 1.0
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
SCHEDULER_MODE = 'min'

# ============================================================================
# LOSS FUNCTION CONFIGURATION
# ============================================================================
LOSS_ALPHA_HEATMAP = 1.0
LOSS_ALPHA_OFFSET = 0.5
LOSS_ALPHA_SPATIAL = 0.3
LOSS_MIN_DISTANCE = 20
LOSS_FOCAL_ALPHA = 2.0
LOSS_FOCAL_GAMMA = 4.0
LOSS_OFFSET_CONFIDENCE_THRESHOLD = 0.1

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
HEATMAP_THRESHOLD = 0.6
SPATIAL_THRESHOLD = 3.0
MIN_DISTANCE = 10
DEVICE = 'cuda'
QUANTIZE = False

# ============================================================================
# VIDEO INFERENCE CONFIGURATION
# ============================================================================
SKIP_FRAMES = 0

# ============================================================================
# EDGE OPTIMIZATION CONFIGURATION
# ============================================================================
USE_QUANTIZATION = False
REDUCE_RESOLUTION = False
QUANTIZATION_BACKEND = 'fbgemm'

# ============================================================================
# CHECKPOINT CONFIGURATION
# ============================================================================
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_SAVE_INTERVAL = 10
CHECKPOINT_KEEP_BEST_ONLY = True
CHECKPOINT_FILENAME = 'best_model.pt'
HISTORY_FILENAME = 'history.json'

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
EVAL_HEATMAP_THRESHOLD = 0.6
EVAL_SPATIAL_THRESHOLD = 3.0
EVAL_MIN_DISTANCE = 10
EVAL_BATCH_SIZE = 16
EVAL_NUM_VISUALIZATION_SAMPLES = 5
VISUALIZATION_DIR = 'visualizations'

# ============================================================================
# DATA AUGMENTATION CONFIGURATION
# ============================================================================
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.2
AUG_SATURATION = 0.1
AUG_ROTATION_DEGREES = 5

# ============================================================================
# DERIVED CONFIGURATION (Do not modify)
# ============================================================================
# Actual model output with stride architecture:
# 416×312 → ... → 13×10 (not 13×9.75)
HMAP_HEIGHT = 10  # 312 / 32 ≈ 10 (rounded by pooling)
HMAP_WIDTH = 13   # 416 / 32 ≈ 13