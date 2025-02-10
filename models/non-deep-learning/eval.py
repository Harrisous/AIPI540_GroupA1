import cv2
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pathlib
import json
from train import extract_hog_features

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()
ROOT_DIR = CURRENT_DIR.parent.parent

def load_evaluation_data(dataset_path):
    features = []
    labels = []
    names = []  # Store actual names for better reporting
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
            names.append(person)
    
    return np.array(features), np.array(labels), np.array(names), label_dict

def evaluate_models(dataset_path, 
                   svm_path="face_svm_model.joblib",
                   rf_path="face_rf_model.joblib"):
    """
    Evaluate both SVM and Random Forest models
    """
    # Load models
    svm_model = joblib.load(svm_path)
    rf_model = joblib.load(rf_path)
    
    # Load evaluation data
    print("Loading and preparing evaluation data...")
    features, labels, names, label_dict = load_evaluation_data(dataset_path)
    
    # Invert label dictionary for readable output
    inv_label_dict = {v: k for k, v in label_dict.items()}
    class_names = [inv_label_dict[i] for i in range(len(label_dict))]
    
    # 创建结果字典
    results = {
        'dataset_info': {
            'total_samples': len(features),
            'num_classes': len(label_dict),
            'classes': list(label_dict.keys())
        },
        'models': {}
    }
    
    # Evaluate SVM
    print("\nEvaluating SVM Model:")
    svm_pred = svm_model.predict(features)
    svm_accuracy = accuracy_score(labels, svm_pred)
    print(f"SVM Accuracy: {svm_accuracy:.2%}")
    svm_report = classification_report(labels, svm_pred, 
                                     target_names=class_names,
                                     output_dict=True)
    results['models']['svm'] = {
        'accuracy': float(svm_accuracy),
        'classification_report': svm_report
    }
    print("\nSVM Classification Report:")
    print(classification_report(labels, svm_pred, target_names=class_names))
    
    # Evaluate Random Forest
    print("\nEvaluating Random Forest Model:")
    rf_pred = rf_model.predict(features)
    rf_accuracy = accuracy_score(labels, rf_pred)
    print(f"Random Forest Accuracy: {rf_accuracy:.2%}")
    rf_report = classification_report(labels, rf_pred, 
                                    target_names=class_names,
                                    output_dict=True)
    results['models']['random_forest'] = {
        'accuracy': float(rf_accuracy),
        'classification_report': rf_report
    }
    print("\nRandom Forest Classification Report:")
    print(classification_report(labels, rf_pred, target_names=class_names))
    
    # Evaluate Ensemble
    print("\nEvaluating Ensemble Model (Confidence-based):")
    ensemble_predictions = []
    for i in range(len(features)):
        svm_prob = svm_model.predict_proba([features[i]])[0]
        rf_prob = rf_model.predict_proba([features[i]])[0]
        
        svm_conf = np.max(svm_prob)
        rf_conf = np.max(rf_prob)
        
        if svm_conf > rf_conf:
            ensemble_predictions.append(svm_pred[i])
        else:
            ensemble_predictions.append(rf_pred[i])
    
    ensemble_accuracy = accuracy_score(labels, ensemble_predictions)
    print(f"Ensemble Accuracy: {ensemble_accuracy:.2%}")
    ensemble_report = classification_report(labels, ensemble_predictions, 
                                         target_names=class_names,
                                         output_dict=True)
    results['models']['ensemble'] = {
        'accuracy': float(ensemble_accuracy),
        'classification_report': ensemble_report
    }
    print("\nEnsemble Classification Report:")
    print(classification_report(labels, ensemble_predictions, target_names=class_names))
    
    # Calculate per-person accuracy
    per_person_results = {}
    for person in label_dict:
        person_idx = labels == label_dict[person]
        person_total = np.sum(person_idx)
        if person_total > 0:
            svm_correct = np.sum((svm_pred == labels) & person_idx)
            rf_correct = np.sum((rf_pred == labels) & person_idx)
            ensemble_correct = np.sum((ensemble_predictions == labels) & person_idx)
            
            per_person_results[person] = {
                'total_samples': int(person_total),
                'svm': {
                    'correct': int(svm_correct),
                    'accuracy': float(svm_correct/person_total)
                },
                'random_forest': {
                    'correct': int(rf_correct),
                    'accuracy': float(rf_correct/person_total)
                },
                'ensemble': {
                    'correct': int(ensemble_correct),
                    'accuracy': float(ensemble_correct/person_total)
                }
            }
            
            print(f"\n{person}:")
            print(f"  SVM: {svm_correct}/{person_total} ({svm_correct/person_total:.2%})")
            print(f"  RF:  {rf_correct}/{person_total} ({rf_correct/person_total:.2%})")
            print(f"  Ensemble: {ensemble_correct}/{person_total} ({ensemble_correct/person_total:.2%})")
    
    results['per_person_accuracy'] = per_person_results
    
    # 保存结果到JSON文件
    results_path = CURRENT_DIR / "model" / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nEvaluation results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    dataset_path = ROOT_DIR / "data/processed/test"
    model_dir = CURRENT_DIR / "model"
    svm_path = model_dir / "face_svm_model.joblib"
    rf_path = model_dir / "face_rf_model.joblib"
    
    evaluate_models(dataset_path, svm_path, rf_path)
