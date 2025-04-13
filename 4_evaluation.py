import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from config import DATA_PATH, FEATURE_PATH, MODEL_PATH

def evaluate():
    # Load data
    model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'fusion_model.h5'))
    cnn_features = np.load(os.path.join(FEATURE_PATH, 'cnn_features.npy'))
    dip_features = pd.read_csv(os.path.join(FEATURE_PATH, 'dip_features.csv'))
    labels = pd.read_csv(os.path.join(DATA_PATH, 'metadata.csv'))['label']
    
    # Prepare features
    scaler = StandardScaler()
    dip_scaled = scaler.fit_transform(dip_features)
    
    # Evaluate
    y_pred = model.predict([cnn_features, dip_scaled])
    y_pred = (y_pred > 0.5).astype(int)
    
    print(classification_report(labels, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(labels, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    evaluate()