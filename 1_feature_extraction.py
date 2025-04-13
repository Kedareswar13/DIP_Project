import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
from config import DATA_PATH, PROCESSED_PATH, FEATURE_PATH, QUALITY
from utils.dip_utils import DIPFeatureExtractor

def process_images():
    """Resave images with compression and create metadata"""
    dip = DIPFeatureExtractor(visualize=True)
    metadata = []
    dip_features = []

    for label in ['Au', 'Tp']:
        class_path = os.path.join(DATA_PATH, label)
        images = [img for img in os.listdir(class_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in tqdm(images, desc=f"Processing {label}"):
            
            orig_path = os.path.join(class_path, img_name)
            proc_path = os.path.join(PROCESSED_PATH, f"{label}_{os.path.splitext(img_name)[0]}.jpg")
            
            try:
                # Resave with compression
                img = Image.open(orig_path).convert('RGB')
                img.save(proc_path, 'JPEG', quality=QUALITY)
                
                # Extract features with class label
                features = dip.extract_features(orig_path, proc_path, label)
                
                dip_features.append(features)
                metadata.append({
                    'original_path': orig_path,
                    'processed_path': proc_path,
                    'label': 0 if label == 'Au' else 1
                })
            except Exception as e:
                print(f"Error processing {orig_path}: {str(e)}")

    # Save data
    df_meta = pd.DataFrame(metadata)
    df_dip = pd.DataFrame(dip_features)
    
    df_meta.to_csv(os.path.join(DATA_PATH, 'metadata.csv'), index=False)
    df_dip.to_csv(os.path.join(FEATURE_PATH, 'dip_features.csv'), index=False)
    
    print(f"Processed {len(df_meta)} images. Metadata and features saved.")
    print("Generated visualizations for 5 samples from each class in:")
    print(f"- {os.path.abspath(os.path.join(dip.output_dir, 'Au'))}")
    print(f"- {os.path.abspath(os.path.join(dip.output_dir, 'Tp'))}")

if __name__ == '__main__':
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    os.makedirs(FEATURE_PATH, exist_ok=True)
    process_images()