import os
import cv2
import joblib
import numpy as np

from src.detection import load_model as load_detector, detect_vehicles
from src.hsv import extract_normalized_hsv_histogram, extract_lbp_histogram

# ─── Configuration ─────────────────────────────────────────────────────────
# PROJECT_ROOT    = os.path.abspath(os.path.dirname(__file__))

PROJECT_ROOT     = os.path.dirname(os.path.dirname(__file__))  # Go up from src/
MODEL_DIR        = os.path.join(PROJECT_ROOT, "model")
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Paths to your saved models/scalers
DETECTOR_MODEL_PATH = None    # load_model() in detection.py knows its own defaults
CLASSIFIER_PATH  = os.path.join(MODEL_DIR, "rf_combined_final.pkl")
SCALER_PATH      = os.path.join(MODEL_DIR, "hog_scaler.pkl")

# Vehicle class names (must match your label encoding)
CLASS_NAMES = ["cars", "trucks", "threewheel", "motorcycle", "non-vehicle"]


# ─── Initialization ────────────────────────────────────────────────────────
# 1) Detection model (HOG+SVM)
detector_model, detector_scaler = load_detector()

# 2) Classification model & scaler (HSV+LBP + RF)
rf          = joblib.load(CLASSIFIER_PATH)
feat_scaler = joblib.load(SCALER_PATH)


# ─── Core Functions ────────────────────────────────────────────────────────

def classify_crop(img_crop: np.ndarray) -> str:
    """
    Given a BGR crop, extract HSV+LBP features, scale them,
    predict with RF, and return class name.
    """
    # 1) HSV histogram
    hsv_feat = extract_normalized_hsv_histogram(img_crop)
    # 2) LBP histogram
    lbp_feat = extract_lbp_histogram(img_crop)
    # 3) combine & scale
    fv = np.hstack([hsv_feat, lbp_feat]).reshape(1, -1)
    fv_s = feat_scaler.transform(fv)
    # 4) predict
    lbl_idx = rf.predict(fv_s)[0]
    return CLASS_NAMES[int(lbl_idx)]


def run_pipeline(image_path: str, threshold: float = 1.0, visualize: bool = True):
    """
    Full detection -> classification on one image.
    Returns annotated image (BGR).
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")
    orig = img.copy()

    # 1) Detect vehicles
    dets = detect_vehicles(img, detector_model, detector_scaler, threshold=threshold)

    # 2) For each detection, classify and annotate
    for (x1, y1, x2, y2, det_lbl, score) in dets:
        crop = orig[y1:y2, x1:x2]
        cls  = classify_crop(crop)

        # Draw box & text
        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{cls} {score*100:.1f}%"
        cv2.putText(
            orig, text, (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        )

    # 3) Save &/or display
    out_path = os.path.join(RESULTS_DIR, os.path.basename(image_path))
    cv2.imwrite(out_path, orig)
    if visualize:
        cv2.imshow("Detection + Classification", orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"✅ Results saved to {out_path}")
    return orig


# ─── Run as script ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Full pipeline: HOG+SVM detection → RF classification"
    )
    p.add_argument("image", help="Path to input image")
    p.add_argument(
        "--thresh", type=float, default=1.0,
        help="SVM decision threshold for detection"
    )
    p.add_argument(
        "--no-display", action="store_true",
        help="Do not open a window to display results"
    )
    args = p.parse_args()

    run_pipeline(
        args.image,
        threshold=args.thresh,
        visualize=not args.no_display
    )