import cv2
import numpy as np
import os
import pathlib

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()
ROOT_DIR = CURRENT_DIR.parent.parent


def load_training_data(dataset_path):
    images = []
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
            images.append(img)
            labels.append(label_dict[person])

    return images, np.array(labels), label_dict


def train_lbph(
    dataset_path, model_path="lbph_face_model.xml", label_map_path="label_map.npy"
):
    faces, labels, label_map = load_training_data(dataset_path)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)
    recognizer.save(model_path)

    print("Training complete. Model saved to:", model_path)
    np.save(label_map_path, label_map)  # Save label mapping for inference


if __name__ == "__main__":
    dataset_path = ROOT_DIR / "data/raw"
    model_path = CURRENT_DIR / "model/lbph_face_model.xml"
    label_map_path = CURRENT_DIR / "model/label_map.npy"
    train_lbph(dataset_path, model_path, label_map_path)
