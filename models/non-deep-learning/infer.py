import cv2
import numpy as np
import sys
import pathlib
import joblib
from train import extract_hog_features

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()
ROOT_DIR = CURRENT_DIR.parent.parent

def load_models(svm_path="face_svm_model.joblib", 
               rf_path="face_rf_model.joblib",
               label_map_path="label_map.npy"):
    """
    Load both SVM and Random Forest models along with the label mapping
    """
    svm_model = joblib.load(svm_path)
    rf_model = joblib.load(rf_path)
    label_map = np.load(label_map_path, allow_pickle=True).item()
    
    # Invert label map for prediction (number to name)
    inv_label_map = {v: k for k, v in label_map.items()}
    return svm_model, rf_model, inv_label_map

def recognize_face(image_path, model_dir=None):
    """
    Recognize face using both SVM and Random Forest models
    Args:
        image_path: str or pathlib.Path, path to the image file
        model_dir: str or pathlib.Path, directory containing the models (optional)
    Returns:
        tuple: (predicted_name, confidence)
    """
    if model_dir is None:
        model_dir = CURRENT_DIR / "model"
    else:
        model_dir = pathlib.Path(model_dir)
    
    # Load models
    svm_path = model_dir / "face_svm_model.joblib"
    rf_path = model_dir / "face_rf_model.joblib"
    label_map_path = model_dir / "label_map.npy"
    
    svm_model, rf_model, inv_label_map = load_models(svm_path, rf_path, label_map_path)
    
    # Load and preprocess image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Error: Image not found!")
    
    # Extract HOG features
    hog_features = extract_hog_features(img)
    hog_features = hog_features.reshape(1, -1)  # Reshape for prediction
    
    # Get predictions from both models
    svm_pred = svm_model.predict(hog_features)[0]
    svm_prob = svm_model.predict_proba(hog_features).max()
    
    rf_pred = rf_model.predict(hog_features)[0]
    rf_prob = rf_model.predict_proba(hog_features).max()
    
    # Get predicted names
    svm_name = inv_label_map.get(svm_pred, "Unknown")
    rf_name = inv_label_map.get(rf_pred, "Unknown")
    
    # Return consensus if both models agree, otherwise return prediction with higher confidence
    if svm_name == rf_name:
        final_prediction = svm_name
        confidence = max(svm_prob, rf_prob)
    else:
        final_prediction = svm_name if svm_prob > rf_prob else rf_name
        confidence = max(svm_prob, rf_prob)
    
    return final_prediction, confidence

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = CURRENT_DIR / "test_image.jpg"
    
    try:
        name, conf = recognize_face(image_path)
        print(f"\nPrediction Result:")
        print(f"Name: {name}")
        print(f"Confidence: {conf:.2%}")
    except Exception as e:
        print(f"Error: {str(e)}")
