import cv2
import numpy as np
import os
import pathlib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()
ROOT_DIR = CURRENT_DIR.parent.parent

def extract_hog_features(image):
    # HOG parameters
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    # Resize image to match window size
    image = cv2.resize(image, win_size)
    
    # Initialize HOG descriptor
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    # Compute HOG features
    hog_features = hog.compute(image)
    return hog_features.flatten()

def load_training_data(dataset_path):
    features = []
    labels = []
    label_dict = {}  # Mapping person names to numeric labels
    label_id = 0

    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue

        if person not in label_dict:
            label_dict[person] = label_id
            label_id += 1

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Extract HOG features
            hog_features = extract_hog_features(img)
            features.append(hog_features)
            labels.append(label_dict[person])

    return np.array(features), np.array(labels), label_dict

def train_model(dataset_path, model_dir="model"):
    # Create model directory if it doesn't exist
    model_dir = CURRENT_DIR / model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths for saving models and label map
    svm_path = model_dir / "face_svm_model.joblib"
    rf_path = model_dir / "face_rf_model.joblib"
    label_map_path = model_dir / "label_map.npy"
    
    # Load and prepare data
    print("Loading and preparing training data...")
    features, labels, label_map = load_training_data(dataset_path)
    
    # Train Random Forest
    print("Training Random Forest classifier...")
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=3, n_jobs=-1)
    rf_grid.fit(features, labels)
    print("Best Random Forest parameters:", rf_grid.best_params_)
    
    # Train SVM with grid search
    print("Training SVM classifier...")
    svm_params = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.001, 0.01]
    }
    svm = SVC(probability=True, random_state=42)
    svm_grid = GridSearchCV(svm, svm_params, cv=3, n_jobs=-1)
    svm_grid.fit(features, labels)
    print("Best SVM parameters:", svm_grid.best_params_)
    
    # Save models and label map
    joblib.dump(svm_grid.best_estimator_, svm_path)
    joblib.dump(rf_grid.best_estimator_, rf_path)
    np.save(label_map_path, label_map)
    
    print("Training complete. Models saved to:", model_dir)
    return svm_grid.best_estimator_, rf_grid.best_estimator_, label_map

if __name__ == "__main__":
    dataset_path = ROOT_DIR / "data/processed/train"  # 修改为使用processed/train目录
    train_model(dataset_path)
