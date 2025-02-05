import cv2
import numpy as np
import sys

import pathlib

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()
ROOT_DIR = CURRENT_DIR.parent.parent

def load_model(model_path="lbph_face_model.xml", label_map_path="label_map.npy"):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    label_map = np.load(label_map_path, allow_pickle=True).item()
    return recognizer, label_map

def recognize_face(image_path, model_path="lbph_face_model.xml", label_map_path="label_map.npy"):
    recognizer, label_map = load_model(model_path, label_map_path)
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found!")
        return
    
    label, confidence = recognizer.predict(img)
    predicted_person = label_map.get(label, "Unknown")
    print(f"Predicted Person: {predicted_person}, Confidence: {confidence}")

if __name__ == "__main__":
    image_path = CURRENT_DIR / "IMG_3489.jpg"
    model_path = CURRENT_DIR / "model/lbph_face_model.xml"
    label_map_path = CURRENT_DIR / "model/label_map.npy"
    recognize_face(image_path, model_path, label_map_path)
