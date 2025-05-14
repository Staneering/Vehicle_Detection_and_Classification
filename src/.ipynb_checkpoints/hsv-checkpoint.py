import cv2
import os
import numpy as np
from typing import List

from tqdm import tqdm
# from src.hsv import extract_normalized_hsv_histogram


# from src.hsv import extract_normalized_hsv_histogram
from src.hpy import extract_lbp_histogram  # the function above
from skimage.feature import local_binary_pattern


def extract_hsv_histogram(image: np.ndarray, bins: List[int] = [8, 8, 8]) -> np.ndarray:
    """
    Extract a normalized HSV color histogram from an image.

    Parameters:
    - image: np.ndarray — Input image in BGR format (as loaded by cv2.imread).
    - bins: list[int] — Number of bins for H, S, and V channels respectively.

    Returns:
    - np.ndarray — Flattened and normalized histogram as feature vector.
    """
    # Convert BGR image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Compute 3D HSV histogram
    hist = cv2.calcHist(
        [hsv_image],          # Input image
        [0, 1, 2],            # Channels: H, S, V
        None,                 # No mask
        bins,                 # Histogram bin sizes for H, S, V
        [0, 180, 0, 256, 0, 256]  # Ranges for H, S, V
    )

    # Normalize the histogram and flatten it into a 1D feature vector
    hist = cv2.normalize(hist, hist).flatten()

    return hist


def extract_normalized_hsv_histogram(image: np.ndarray, bins: List[int] = [8, 8, 8]) -> np.ndarray:
    """
    Extract and normalize HSV histogram so that sum = 1 (scale-invariant).
    """
    hist = extract_hsv_histogram(image, bins)
    hist /= (hist.sum() + 1e-6)  # Manual normalization
    return hist



def load_images_from_folder(folder_path, label, bins=(8, 8, 8)):
    """
    Recursively load all images under folder_path (including subfolders),
    extract HSV histograms, and assign a single label to all of them.
    """
    features, labels = [], []
    # Walk through all subdirectories
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            file_path = os.path.join(root, filename)
            image = cv2.imread(file_path)
            if image is None:
                continue
            hist = extract_normalized_hsv_histogram(image, bins)
            features.append(hist)
            labels.append(label)
    return features, labels

def extract_batch_features(data_root="Data", bins=(8,8,8)):
    vehicle_dir     = os.path.join(data_root, "vehicle")
    non_vehicle_dir = os.path.join(data_root, "non-vehicle")

    # vehicle = label 1, non-vehicle = label 0
    vf, vl = load_images_from_folder(vehicle_dir,     label=1, bins=bins)
    nf, nl = load_images_from_folder(non_vehicle_dir, label=0, bins=bins)
    
    all_features = np.array(vf + nf)
    all_labels   = np.array(vl + nl)
    return all_features, all_labels

if __name__ == "__main__":
    feats, labs = extract_batch_features(data_root="Data")
    os.makedirs("Data/splits", exist_ok=True)
    np.save("Data/splits/hsv_features.npy", feats)
    np.save("Data/splits/hsv_labels.npy",   labs)
    print(f"Saved {len(feats)} feature vectors and labels to Data/splits/")



def extract_lbp_histogram(image: np.ndarray,
                          P: int = 8,
                          R: float = 1.0,
                          bins: int = 256) -> np.ndarray:
    """
    Extract a normalized LBP histogram from a grayscale image.

    Args:
      image: BGR image as np.ndarray.
      P: number of circularly symmetric neighbor set points.
      R: radius of circle.
      bins: number of histogram bins (typically 256 for LBP).

    Returns:
      1D np.ndarray of length `bins`, normalized to sum=1.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute LBP image
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    # Build histogram
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=bins,
                             range=(0, bins))
    # Normalize
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist



def extract_batch_hsv_lbp(data_root="Data"):
    veh = os.path.join(data_root, "vehicle")
    nonveh = os.path.join(data_root, "non-vehicle")
    X_color, y_color = [], []
    X_lbp,   y_lbp   = [], []
    
    # Helper to walk folders
    def process(folder, label):
        for root, _, files in os.walk(folder):
            for f in files:
                if not f.lower().endswith((".jpg","jpeg","png")):
                    continue
                path = os.path.join(root, f)
                img = cv2.imread(path)
                if img is None: 
                    continue
                X_color.append(extract_normalized_hsv_histogram(img))
                X_lbp.append(extract_lbp_histogram(img))
                y_color.append(label)
                y_lbp.append(label)
    
    process(veh,     1)
    process(nonveh,  0)
    
    return (np.array(X_color), np.array(X_lbp), np.array(y_color))

if __name__ == "__main__":
    X_hsv, X_lbp, y = extract_batch_hsv_lbp(data_root="Data")
    # Feature fusion:
    X_combined = np.hstack([X_hsv, X_lbp])
    # Save
    np.save("Data/splits/combined_features.npy", X_combined)
    np.save("Data/splits/labels.npy", y)
    print("Saved combined HSV+LBP features and labels.")


