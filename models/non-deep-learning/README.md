# Face Recognition with HOG + SVM/RF

This directory contains the implementation of face recognition using HOG (Histogram of Oriented Gradients) features with SVM (Support Vector Machine) and Random Forest classifiers.

## Model Architecture

- Feature Extraction: HOG (Histogram of Oriented Gradients)
- Classifiers: 
  - SVM (Support Vector Machine)
  - Random Forest
  - Ensemble (Confidence-based voting between SVM and RF)

## Directory Structure

```
models/non-deep-learning/
├── train.py           # Training script
├── eval.py            # Evaluation script
├── infer.py           # Inference script for single image
└── model/             # Directory for saved models and results
    ├── face_svm_model.joblib      # Trained SVM model
    ├── face_rf_model.joblib       # Trained Random Forest model
    ├── label_map.npy              # Label mapping
    └── evaluation_results.json    # Evaluation metrics
```

## Usage

### 1. Training

Train the SVM and Random Forest models:

```bash
python train.py
```

This will:
- Load training data from `data/processed/train`
- Extract HOG features from face images
- Train SVM and Random Forest models with grid search
- Save the models in the `model` directory

### 2. Evaluation

Evaluate the trained models on the test set:

```bash
python eval.py
```

This will:
- Load test data from `data/processed/test`
- Evaluate both SVM and Random Forest models
- Evaluate the ensemble model
- Print detailed performance metrics
- Save evaluation results to `model/evaluation_results.json`

### 3. Inference

There are two ways to use the inference script:

#### Command Line Usage
```bash
python infer.py path/to/your/image.jpg
```

If no image path is provided, it will use a default test image.

#### Programmatic Usage
```python
from infer import recognize_face

# Using default model directory
name, confidence = recognize_face("path/to/image.jpg")

# Or specify a custom model directory
name, confidence = recognize_face("path/to/image.jpg", model_dir="path/to/model_dir")

print(f"Predicted: {name} with confidence: {confidence:.2%}")
```

Returns:
- `name`: Predicted person's name (str)
- `confidence`: Prediction confidence score (float between 0 and 1)

## Evaluation Metrics

The evaluation results (`evaluation_results.json`) include:
- Dataset information
- Overall accuracy for each model
- Detailed classification reports
- Per-person accuracy statistics

## Model Parameters

### HOG Parameters
- Window size: 64x64
- Block size: 16x16
- Block stride: 8x8
- Cell size: 8x8
- Number of bins: 9

### SVM Parameters (Grid Search)
- C: [0.1, 1, 10, 100]
- Kernel: ['rbf', 'linear']
- Gamma: ['scale', 'auto', 0.001, 0.01]

### Random Forest Parameters (Grid Search)
- n_estimators: [100, 200]
- max_depth: [None, 10, 20]
- min_samples_split: [2, 5] 