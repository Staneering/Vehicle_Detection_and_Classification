import os
import glob
import cv2
import numpy as np
from skimage.feature import hog
from .utils import load_and_sample_images

def fetch_and_sample_image_paths(
    vehicle_dir: str,
    non_vehicle_dir: str,
    sample_size: int = 300
):
    """
    Returns a list of (img_path, label) pairs:
     - For each subfolder under vehicle_dir, samples up to `sample_size` images
       and labels them by subfolder name.
     - For non_vehicle_dir, samples up to `sample_size` images and labels them
       as 'non-vehicle'.
    """
    annotations = []

    # Vehicle subfolders
    for folder in os.listdir(vehicle_dir):
        folder_path = os.path.join(vehicle_dir, folder)
        if os.path.isdir(folder_path):
            imgs = load_and_sample_images(folder_path, sample_size)
            annotations.extend((p, folder) for p in imgs)

    # Non-vehicle folder
    nonveh = load_and_sample_images(non_vehicle_dir, sample_size)
    annotations.extend((p, "non-vehicle") for p in nonveh)

    return annotations

def preprocess_image(img_path: str, size=(128, 128)) -> np.ndarray:
    """
    Reads, resizes, converts to gray, applies CLAHE, and returns the processed image.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}")
    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def extract_hog_features(img: np.ndarray) -> np.ndarray:
    """
    Takes a single-channel (grayscale) image, ensures 128×128 size,
    and returns a 1D HOG feature vector.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    feats = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )
    return feats

def load_and_extract_features(annotations):
    """
    Given annotations = list of (img_path, label), returns:
      X: np.ndarray of shape (N, D) of HOG features
      y: np.ndarray of shape (N,) of labels
    """
    features, labels = [], []
    for img_path, label in annotations:
        try:
            proc = preprocess_image(img_path)
            feat = extract_hog_features(proc)
            features.append(feat)
            labels.append(label)
        except Exception as e:
            print(f"⚠️ Skipping {img_path}: {e}")
    return np.array(features), np.array(labels)