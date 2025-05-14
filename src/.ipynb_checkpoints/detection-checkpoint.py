import os
import cv2
import joblib
import numpy as np

# from src.detection import load_model as load_detector, detect_vehicles
from src.preprocessing import (
    extract_normalized_hsv_histogram,
    extract_lbp_histogram,
    extract_hog_features
)

# ─── Configuration ─────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "model")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASSIFIER_DIR = MODEL_DIR

# Classifier and scaler paths
CLASSIFIER_PATHS = {
    "hog":     os.path.join(CLASSIFIER_DIR, "vehicle_svm_model.pkl"),
    "hog_hsv": os.path.join(CLASSIFIER_DIR, "rf_combined_final.pkl"),
    "hsv_lbp": os.path.join(CLASSIFIER_DIR, "rf_combined_final.pkl")
}

SCALER_PATHS = {
    "hog":     os.path.join(CLASSIFIER_DIR, "hog_scaler.pkl"),
    "hog_hsv": os.path.join(CLASSIFIER_DIR, "cfeature_scaler.pkl"),
    "hsv_lbp": os.path.join(CLASSIFIER_DIR, "hsv_lbp_scaler.pkl")
}

CLASS_NAMES = ["cars", "trucks", "threewheel", "motorcycle", "non-vehicle"]

# ─── Initialization ────────────────────────────────────────────────────────
detector_model, detector_scaler = load_model()

# ─── Feature Modes ─────────────────────────────────────────────────────────
def extract_features_by_mode(crop, mode):
    if mode == "hsv_lbp":
        hsv_feat = extract_normalized_hsv_histogram(crop)
        lbp_feat = extract_lbp_histogram(crop)
        return np.hstack([hsv_feat, lbp_feat])
    elif mode == "hog":
        return extract_hog_features(crop)
    elif mode == "hog_hsv":
        hog_feat = extract_hog_features(crop)
        hsv_feat = extract_normalized_hsv_histogram(crop)
        return np.hstack([hog_feat, hsv_feat])
    else:
        raise ValueError(f"Unknown mode: {mode}")

def load_classifier_and_scaler(mode):
    clf = joblib.load(CLASSIFIER_PATHS[mode])
    scaler = joblib.load(SCALER_PATHS[mode])
    return clf, scaler

# ─── Classification Function ───────────────────────────────────────────────
def classify_crop(img_crop, clf, scaler, mode):
    """
    Classifies a cropped image region using the trained classifier and feature extractor.
    """
    # Resize crop to match training size (IMPORTANT!)
    resized_crop = cv2.resize(img_crop, (128, 128))  # ← Make sure 128x128 matches your training size

    # Extract features and scale
    fv = extract_features_by_mode(resized_crop, mode).reshape(1, -1)
    fv_scaled = scaler.transform(fv)

    # Predict class index
    lbl_idx = clf.predict(fv_scaled)[0]
    return CLASS_NAMES[int(lbl_idx)]

# ─── Pipeline Execution ────────────────────────────────────────────────────
def run_pipeline(image_path: str, mode: str = "hsv_lbp", threshold: float = 1.0, visualize: bool = True):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"❌ Could not load image at: {image_path}")
    
    orig = img.copy()

    # Load classifier and scaler
    clf, scaler = load_classifier_and_scaler(mode)

    # Step 1: Vehicle Detection
    dets = detect_vehicles(img, detector_model, detector_scaler, threshold=threshold)

    # Step 2: Classification
    for (x1, y1, x2, y2, det_lbl, score) in dets:
        crop = orig[y1:y2, x1:x2]
        cls = classify_crop(crop, clf, scaler, mode)

        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{cls} {score*100:.1f}%"
        cv2.putText(orig, label_text, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Step 3: Save/Display result
    out_path = os.path.join(RESULTS_DIR, os.path.basename(image_path))
    cv2.imwrite(out_path, orig)
    
    if visualize:
        cv2.imshow("Detection + Classification", orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"✅ Results saved to: {out_path}")
    return orig

# ─── CLI Entry Point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Vehicle Detection and Classification Pipeline")
    p.add_argument("image", help="Path to input image")
    p.add_argument("--mode", choices=["hsv_lbp", "hog", "hog_hsv"], default="hsv_lbp", help="Feature mode")
    p.add_argument("--thresh", type=float, default=1.0, help="Detection threshold")
    p.add_argument("--no-display", action="store_true", help="Suppress image display")
    args = p.parse_args()

    run_pipeline(args.image, mode=args.mode, threshold=args.thresh, visualize=not args.no_display)