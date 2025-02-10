import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt

# Configuration
DATASET_PATH = "data/processed"
TARGET_SIZE = (100, 100)
N_COMPONENTS = 50  # Number of principal components for PCA

def detect_eyes(image):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    return eyes if eyes is not None and len(eyes) >= 2 else None

def align_face(image, eyes):
    if eyes is None or len(eyes) < 2:
        return cv2.resize(image, TARGET_SIZE)
    
    eyes = sorted(eyes, key=lambda x: x[0])[:2]
    (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes
    
    left_center = (int(x1 + w1/2), int(y1 + h1/2))
    right_center = (int(x2 + w2/2), int(y2 + h2/2))
    
    dx = right_center[0] - left_center[0]
    dy = right_center[1] - left_center[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    center = (int((left_center[0] + right_center[0])/2),
              int((left_center[1] + right_center[1])/2))
    
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return cv2.resize(rotated, TARGET_SIZE)

def preprocess(image):
    if image is None:
        raise ValueError("Input image is None")
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    eyes = detect_eyes(gray)
    aligned = align_face(gray, eyes)
    equalized = cv2.equalizeHist(aligned)
    return equalized.astype(np.float32) / 255.0

def load_data():
    images = {'train': [], 'val': [], 'test': []}
    labels = {'train': [], 'val': [], 'test': []}
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(DATASET_PATH, split)
        for person in os.listdir(split_path):
            person_dir = os.path.join(split_path, person)
            if os.path.isdir(person_dir):
                for img_file in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, img_file)
                    image = cv2.imread(img_path)
                    if image is not None:
                        processed = preprocess(image)
                        images[split].append(processed)
                        labels[split].append(person)
    
    return (np.array(images['train']), np.array(labels['train']),
            np.array(images['val']), np.array(labels['val']),
            np.array(images['test']), np.array(labels['test']))

def apply_pca(X_train, n_components):
    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_flattened)
    return pca, X_train_pca

def project_to_pca_space(image, pca_model):
    image_flattened = image.flatten().reshape(1, -1)
    return pca_model.transform(image_flattened)

def recognize_face_pca(test_image, pca_model, X_train_pca, y_train):
    test_image_pca = project_to_pca_space(test_image, pca_model)
    distances = euclidean_distances(test_image_pca, X_train_pca)
    closest_index = np.argmin(distances)
    return y_train[closest_index], distances[0, closest_index]

def predict_new_image(img_path, pca_model, X_train_pca, y_train):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Failed to load image")
    
    try:
        processed = preprocess(img)
        pred, dist = recognize_face_pca(processed, pca_model, X_train_pca, y_train)
        return pred, dist
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}") from e

if __name__ == "__main__":
    # Load the dataset
    print("Loading dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    # Apply PCA
    print("Applying PCA...")
    pca_model, X_train_pca = apply_pca(X_train, N_COMPONENTS)
    
    print(f"PCA components: {N_COMPONENTS}")
    print(f"Explained variance ratio: {np.sum(pca_model.explained_variance_ratio_):.2f}")

    # Evaluate on test set
    correct = 0
    for img, true_label in zip(X_test, y_test):
        pred_label, distance = recognize_face_pca(img, pca_model, X_train_pca, y_train)
        if pred_label == true_label:
            correct += 1
        print(f"True: {true_label:<10} Predicted: {pred_label:<10} Distance: {distance:.2f}")
    
    accuracy = correct / len(y_test)
    print(f"\nTest Accuracy: {accuracy:.2%}")

    # Visualization of eigenfaces (optional)
    plt.figure(figsize=(20, 4))
    for i in range(10):  # Display first 10 eigenfaces
        plt.subplot(2, 5, i+1)
        eigenface = pca_model.components_[i].reshape(TARGET_SIZE)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f"Eigenface {i+1}")
    plt.tight_layout()
    plt.show()
