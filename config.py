import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
DATA_PATH = os.path.join(BASE_DIR, 'data/CASIA2')
PROCESSED_PATH = os.path.join(BASE_DIR, 'data/processed')
FEATURE_PATH = os.path.join(BASE_DIR, 'features')
MODEL_PATH = os.path.join(BASE_DIR, 'models')

# Image parameters
IMG_SIZE = (224, 224)  # VGG16 input size
QUALITY = 90           # JPEG compression quality
BATCH_SIZE = 32