import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, filters
import os

class DIPFeatureExtractor:
    def __init__(self, visualize=False, output_dir='dip_visualizations'):
        self.sift = cv2.SIFT_create()
        self.visualize = visualize
        self.output_dir = output_dir
        self.class_counts = {'Au': 0, 'Tp': 0}  # Track samples per class
        if visualize:
            os.makedirs(os.path.join(output_dir, 'Au'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'Tp'), exist_ok=True)

    def _save_plot(self, image, title, filename, class_label):
        """Save visualizations in class-specific subdirectories"""
        output_path = os.path.join(self.output_dir, class_label)
        plt.figure(figsize=(8, 6))
        plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        plt.title(f"{title} ({class_label})")
        plt.axis('off')
        plt.savefig(os.path.join(output_path, filename))
        plt.close()

    def calculate_sharpness(self, image, img_name, class_label):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        if self.visualize and self.class_counts[class_label] < 5:
            self._save_plot(laplacian, 
                          f'Laplacian - Sharpness: {laplacian.var():.2f}',
                          f'{img_name}_sharpness.png',
                          class_label)
        return laplacian.var()

    def calculate_edge_density(self, image, img_name, class_label):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = filters.sobel(gray)
        
        if self.visualize and self.class_counts[class_label] < 5:
            self._save_plot(edges, 
                          f'Edge Density: {np.mean(edges):.4f}',
                          f'{img_name}_edge_density.png',
                          class_label)
        return np.mean(edges)

    def calculate_sift_features(self, image, img_name, class_label):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, _ = self.sift.detectAndCompute(gray, None)
        
        if self.visualize and self.class_counts[class_label] < 5:
            img_kp = cv2.drawKeypoints(gray, kp, None)
            self._save_plot(img_kp, 
                          f'SIFT Keypoints: {len(kp)}',
                          f'{img_name}_sift.png',
                          class_label)
        return len(kp)

    def calculate_noise_level(self, image, img_name, class_label):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise = gray - cv2.medianBlur(gray, 3)
        
        if self.visualize and self.class_counts[class_label] < 5:
            self._save_plot(noise, 
                          f'Noise Level: {np.std(noise):.2f}',
                          f'{img_name}_noise.png',
                          class_label)
        return np.std(noise)

    def calculate_tamper_ratio(self, orig, processed, img_name, class_label):
        diff = cv2.absdiff(orig, processed)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if self.visualize and self.class_counts[class_label] < 5 and len(contours) > 0:
            display_img = orig.copy()
            cv2.drawContours(display_img, contours, -1, (0,255,0), 2)
            self._save_plot(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB),
                          f'Tamper Ratio: {self._calc_tamper_area(orig, contours):.4f}',
                          f'{img_name}_tamper.png',
                          class_label)
        
        return self._calc_tamper_area(orig, contours)

    def _calc_tamper_area(self, orig, contours):
        if not contours: return 0
        max_contour = max(contours, key=cv2.contourArea)
        return cv2.contourArea(max_contour)/(orig.shape[0]*orig.shape[1])

    def extract_features(self, orig_path, proc_path, class_label):
        orig = cv2.imread(orig_path)
        proc = cv2.imread(proc_path)
        img_name = os.path.splitext(os.path.basename(orig_path))[0]

        features = {
            'sharpness': self.calculate_sharpness(orig, img_name, class_label),
            'edge_density': self.calculate_edge_density(orig, img_name, class_label),
            'sift_features': self.calculate_sift_features(orig, img_name, class_label),
            'noise_level': self.calculate_noise_level(orig, img_name, class_label),
            'tamper_ratio': self.calculate_tamper_ratio(orig, proc, img_name, class_label)
        }

        if self.visualize and self.class_counts[class_label] < 5:
            self.class_counts[class_label] += 1

        return features