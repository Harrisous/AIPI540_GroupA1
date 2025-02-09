import cv2
import numpy as np
import sys
import pathlib
import joblib
from train import extract_hog_features  # Reuse HOG feature extraction from train.py

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

def recognize_face(image_path, 
                  svm_path="face_svm_model.joblib",
                  rf_path="face_rf_model.joblib", 
                  label_map_path="label_map.npy"):
    """
    Recognize face using both SVM and Random Forest models
    """
    # Load models
    svm_model, rf_model, inv_label_map = load_models(svm_path, rf_path, label_map_path)
    
    # Load and preprocess image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found!")
        return
    
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
    
    # Print results
    print("\nPrediction Results:")
    print(f"SVM Prediction: {svm_name} (Confidence: {svm_prob:.2%})")
    print(f"Random Forest Prediction: {rf_name} (Confidence: {rf_prob:.2%})")
    
    # Return consensus if both models agree, otherwise return prediction with higher confidence
    if svm_name == rf_name:
        final_prediction = svm_name
        confidence = max(svm_prob, rf_prob)
    else:
        final_prediction = svm_name if svm_prob > rf_prob else rf_name
        confidence = max(svm_prob, rf_prob)
    
    print(f"\nFinal Prediction: {final_prediction} (Confidence: {confidence:.2%})")
    return final_prediction, confidence

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = CURRENT_DIR / "test_image.jpg"
    
    model_dir = CURRENT_DIR / "model"
    svm_path = model_dir / "face_svm_model.joblib"
    rf_path = model_dir / "face_rf_model.joblib"
    label_map_path = model_dir / "label_map.npy"
    
    recognize_face(image_path, svm_path, rf_path, label_map_path)
