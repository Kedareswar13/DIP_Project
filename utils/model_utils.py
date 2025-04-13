import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Dropout, concatenate, Input
from tensorflow.keras.models import Model

from config import IMG_SIZE

def create_fusion_model(dip_feature_count):
    # CNN Feature Processor
    cnn_base = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    cnn_features = tf.keras.layers.GlobalAveragePooling2D()(cnn_base.output)
    
    # DIP Feature Processor
    dip_input = Input(shape=(dip_feature_count,))
    dip_features = Dense(128, activation='relu')(dip_input)
    
    # Fusion Layer
    combined = concatenate([cnn_features, dip_features])
    x = Dense(512, activation='relu')(combined)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=[cnn_base.input, dip_input], outputs=outputs)