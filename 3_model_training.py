import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import DATA_PATH, FEATURE_PATH, MODEL_PATH

def build_fusion_model(input_shape, num_dip_features):
    # CNN input branch
    cnn_input = tf.keras.Input(shape=input_shape, name='cnn_input')
    cnn_features = tf.keras.layers.Dense(256, activation='relu')(cnn_input)
    
    # DIP input branch
    dip_input = tf.keras.Input(shape=(num_dip_features,), name='dip_input')
    dip_features = tf.keras.layers.Dense(128, activation='relu')(dip_input)
    
    # Fusion
    combined = tf.keras.layers.concatenate([cnn_features, dip_features])
    x = tf.keras.layers.Dense(512, activation='relu')(combined)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=[cnn_input, dip_input], outputs=output)
    
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall')])
    return model

def train():
    # Load features
    cnn_features = np.load(os.path.join(FEATURE_PATH, 'cnn_features.npy'))
    dip_features = pd.read_csv(os.path.join(FEATURE_PATH, 'dip_features.csv'))
    labels = pd.read_csv(os.path.join(DATA_PATH, 'metadata.csv'))['label']
    
    # Preprocess features
    scaler = StandardScaler()
    dip_scaled = scaler.fit_transform(dip_features)
    
    # Split data
    (X_cnn_train, X_cnn_test,
     X_dip_train, X_dip_test,
     y_train, y_test) = train_test_split(cnn_features, dip_scaled, labels,
                                        test_size=0.2, stratify=labels,
                                        random_state=42)
    
    # Build and train model
    model = build_fusion_model((cnn_features.shape[1],), dip_scaled.shape[1])
    history = model.fit(
        [X_cnn_train, X_dip_train], y_train,
        validation_data=([X_cnn_test, X_dip_test], y_test),
        epochs=20,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3),
            tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_PATH, 'fusion_model.h5'),
                                              save_best_only=True)
        ]
    )
    return history

if __name__ == '__main__':
    os.makedirs(MODEL_PATH, exist_ok=True)
    train()