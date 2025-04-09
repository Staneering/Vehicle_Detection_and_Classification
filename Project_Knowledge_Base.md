
# Vehicle Detection and Classification System: Project Knowledge Base

## Project Overview

### Goal
The goal of this project is to develop a vehicle detection and classification system using traditional machine learning techniques such as Histogram of Oriented Gradients (HOG) and Support Vector Machines (SVM), and deploy the trained model as a Flask API to serve predictions. The system will detect and classify vehicles from images.

### Key Deliverables
- Working **HOG + SVM-based vehicle detection system**.
- Trained **Random Forest** classifier for vehicle classification.
- **Flask API** endpoint for serving predictions.
- **Documentation** (README, demo video).
- Model **performance metrics** (precision, recall, confusion matrix, etc.).

### Project Timeline
- **Week 1**: Setup, Data Collection, Data Preprocessing.
- **Week 2**: Feature Extraction, Model Training, Initial Evaluation.
- **Week 3**: Model Tuning, Evaluation, Integration.
- **Week 4**: Deployment, Testing, Final Reporting.

---

## Phase Breakdown and Function Definitions

### Phase 1: Data Collection
**Goal**: Download and organize datasets into the correct format.

#### Functions

1. **`download_data(urls: List[str], target_directory: str) -> None`**
   - **Description**: Downloads the datasets from provided URLs and stores them in a target directory.
   - **Inputs**: List of dataset URLs (e.g., `["http://example.com/data1.zip", "http://example.com/data2.zip"]`), target directory path.
   - **Outputs**: Confirmation of successful download, handles errors if download fails.
   - **Edge Cases**: 
     - Invalid URL: Handle gracefully by logging an error message and skipping the download.
     - Network issues: Retry download or notify the user after multiple attempts.

2. **`organize_data(source_directory: str, output_directory: str) -> None`**
   - **Description**: Organizes raw dataset into categories (e.g., cars, trucks, bikes, background).
   - **Inputs**: Path to raw dataset.
   - **Outputs**: Organized dataset in subfolders (cars, trucks, etc.).
   - **Edge Cases**: 
     - Missing or incomplete data: Provide warnings or skip unprocessable files.
     - Corrupted or unsupported file types: Log and skip those files.

---

### Phase 2: Data Preprocessing
**Goal**: Preprocess raw dataset for model input (resize, convert to grayscale, normalize, etc.).

#### Functions

1. **`resize_images(images: List[str], target_size: Tuple[int, int]) -> List[str]`**
   - **Description**: Resizes all images in the input list to the specified target size.
   - **Inputs**: List of image paths, target size (e.g., (64, 64)).
   - **Outputs**: List of resized images.
   - **Example Input/Output**:
     - Input: `["/data/images/car1.jpg", "/data/images/car2.jpg"]`, Target size: `(64, 64)`
     - Output: List of resized image data.
   - **Edge Cases**: 
     - If an image path does not exist, log a warning and skip the image.
     - Handle non-image files (e.g., text or corrupted images) gracefully by skipping or notifying the user.

2. **`convert_to_grayscale(images: List[str]) -> List[str]`**
   - **Description**: Converts input images to grayscale.
   - **Inputs**: List of image paths.
   - **Outputs**: List of grayscale images.
   - **Edge Cases**:
     - Corrupted images: Log and skip problematic files.
     - Ensure proper handling of images that may already be grayscale.

3. **`apply_histogram_equalization(images: List[str]) -> List[str]`**
   - **Description**: Applies histogram equalization to enhance image contrast.
   - **Inputs**: List of grayscale images.
   - **Outputs**: List of contrast-enhanced images.
   - **Edge Cases**:
     - Handle grayscale images that are empty or have low contrast without errors.
  
4. **`split_data(images: List[str], split_ratios: Tuple[float, float, float]) -> Tuple[List[str], List[str], List[str]]`**
   - **Description**: Splits images into training, validation, and test datasets according to specified ratios (e.g., 70% train, 20% validation, 10% test).
   - **Inputs**: List of image paths, split ratios.
   - **Outputs**: Three sets: training, validation, and test data.
   - **Edge Cases**:
     - Ensure that the split ratios sum to 100%.
     - Handle cases where there may not be enough images to split (e.g., too few images for validation or testing).

---

### Phase 3: Feature Extraction
**Goal**: Extract relevant features from images for machine learning models (HOG, color, texture).

#### Functions

1. **`extract_hog_features(images: List[str]) -> List[List[float]]`**
   - **Description**: Extracts Histogram of Oriented Gradients (HOG) features from input images.
   - **Inputs**: List of image paths.
   - **Outputs**: List of HOG feature vectors for each image.
   - **Example Input/Output**:
     - Input: `["/data/images/car1.jpg", "/data/images/car2.jpg"]`
     - Output: HOG feature vectors for each image.
   - **Edge Cases**:
     - Handle errors when images fail to load or are improperly formatted.
     - Consider edge cases like very small or large images, which may affect HOG feature extraction.

2. **`extract_color_features(images: List[str]) -> List[List[float]]`**
   - **Description**: Extracts color features, such as HSV histograms, for classification.
   - **Inputs**: List of image paths.
   - **Outputs**: List of color feature vectors for each image.
   - **Edge Cases**:
     - Handle images that are corrupted or grayscale (for color histograms).
     - Provide feedback if the color space extraction fails.

3. **`extract_texture_features(images: List[str]) -> List[List[float]]`**
   - **Description**: Extracts texture features like Local Binary Patterns (LBP).
   - **Inputs**: List of image paths.
   - **Outputs**: List of texture feature vectors for each image.
   - **Edge Cases**:
     - Handle images that may not have enough texture information (e.g., blank or featureless images).

---

### Phase 4: Model Training
**Goal**: Train machine learning models (SVM, Random Forest) using extracted features.

#### Functions

1. **`train_svm(features: List[List[float]], labels: List[int]) -> SVMModel`**
   - **Description**: Trains an SVM classifier using the provided features and labels.
   - **Inputs**: List of feature vectors, corresponding labels.
   - **Outputs**: Trained SVM model.
   - **Edge Cases**:
     - Handle missing or incorrect feature vectors.
     - Consider potential memory issues for large datasets.

2. **`train_random_forest(features: List[List[float]], labels: List[int]) -> RandomForestModel`**
   - **Description**: Trains a Random Forest classifier using the provided features and labels.
   - **Inputs**: List of feature vectors, corresponding labels.
   - **Outputs**: Trained Random Forest model.
   - **Edge Cases**:
     - Handle empty feature datasets gracefully.

3. **`evaluate_model(model: Model, features: List[List[float]], labels: List[int]) -> Dict[str, float]`**
   - **Description**: Evaluates the trained model using metrics like accuracy, precision, recall.
   - **Inputs**: Trained model, features, and labels.
   - **Outputs**: Dictionary of evaluation metrics (e.g., accuracy, precision, recall).
   - **Edge Cases**:
     - Handle model evaluation with an empty or incomplete dataset.
     - Ensure the metrics are calculated based on correct validation/test data.

---

### Phase 5: Vehicle Detection Pipeline
**Goal**: Implement the vehicle detection pipeline using HOG + SVM and perform sliding window detection.

#### Functions

1. **`apply_sliding_window(image: str, window_size: Tuple[int, int], stride: int) -> List[Tuple[int, int]]`**
   - **Description**: Applies sliding window to input image to detect potential vehicle regions.
   - **Inputs**: Image path, window size (e.g., 64x64), stride (e.g., 16px).
   - **Outputs**: List of coordinates of sliding windows.
   - **Edge Cases**:
     - Handle images that are too small for the specified window size.
     - Ensure proper boundary conditions when the sliding window extends beyond image borders.

2. **`non_maximum_suppression(detections: List[Tuple[int, int]], threshold: float) -> List[Tuple[int, int]]`**
   - **Description**: Removes duplicate detections using non-maximum suppression (NMS).
   - **Inputs**: List of detections, NMS threshold.
   - **Outputs**: List of filtered detections.
   - **Edge Cases**:
     - Handle situations where there are no detections.
     - Ensure proper handling of overlapping detections with very low or high confidence.

---

### Phase 6: Deployment
**Goal**: Deploy the trained model as a Flask API to serve vehicle detection predictions.

#### Functions

1. **`create_flask_app(model: Model) -> FlaskApp`**
   - **Description**: Creates a Flask application and serves the trained model through an API.
   - **Inputs**: Trained model.
   - **Outputs**: Flask app instance with an API endpoint.
   - **Edge Cases**:
     - Ensure proper error handling for failed requests or unavailable model files.

2. **`predict_image(image_path: str) -> Dict[str, float]`**
   - **Description**: Predicts vehicle detection and classification from a given image.
   - **Inputs**: Image path.
   - **Outputs**: Dictionary with prediction results (e.g., vehicle type, confidence).
   - **Edge Cases**:
     - Handle missing or corrupted image files.
     - Ensure that predictions can be made on images of varying sizes and formats.

---

### Additional Notes on Integration and Workflow

1. **Data Flow**: Ensure smooth data handoff between phases. For example, the output of the **preprocessing phase** (e.g., resized, grayscale images) should be appropriately handled and passed to the **feature extraction** phase.

2. **Collaboration Between Phases**: Establish clear responsibilities for each phase:
   - **Preprocessing** should output clean, ready-to-use datasets for **feature extraction**.
   - **Model training** should receive fully prepared features and labels.
   - **Detection pipeline** should utilize the trained models from the **model training** phase.

3. **Testing and Evaluation**: Ensure that every phase has unit tests and validation. This includes testing edge cases (e.g., missing or corrupted data), which should be handled gracefully without breaking the pipeline.

---

