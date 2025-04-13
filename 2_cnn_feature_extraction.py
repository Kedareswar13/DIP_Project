import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications import VGG16
from keras.preprocessing import image
from tqdm import tqdm
from config import DATA_PATH, PROCESSED_PATH, FEATURE_PATH, IMG_SIZE

def extract_cnn_features():
    """Extract CNN features using VGG16"""
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    
    # Load metadata
    df = pd.read_csv(os.path.join(DATA_PATH, 'metadata.csv'))
    
    features = []
    for path in tqdm(df['processed_path'], desc="Extracting CNN features"):
        img = image.load_img(path, target_size=IMG_SIZE)
        x = image.img_to_array(img)
        x = tf.keras.applications.vgg16.preprocess_input(x)
        # Corrected line with proper parenthesis closure
        features.append(model.predict(x[np.newaxis, ...], verbose=0).flatten())  
    
    cnn_features = np.array(features)
    np.save(os.path.join(FEATURE_PATH, 'cnn_features.npy'), cnn_features)
    print(f"CNN features saved with shape {cnn_features.shape}")

if __name__ == '__main__':
    os.makedirs(FEATURE_PATH, exist_ok=True)
    extract_cnn_features()