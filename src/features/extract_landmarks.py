import cv2
import mediapipe as mp
import os
import json
from datasets import load_from_disk
from tqdm import tqdm

# Directory dove sono salvati i dati del dataset e i video
dataset_path = "../../data/raw/emosign_dataset"
video_dir = "../../data/raw/videos"  # Cartella con i video scaricati
output_dir = "../../data/interim/landmarks/"
os.makedirs(output_dir, exist_ok=True)

# Carica il dataset
dataset = load_from_disk(dataset_path)

# Inizializza Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5
)

# Processa il dataset video
for split in dataset.keys():
    split_output_dir = os.path.join(output_dir, split)
    os.makedirs(split_output_dir, exist_ok=True)

    print(f"Processing split: {split}")
    for example in tqdm(dataset[split], desc=f"Extracting landmarks for {split}"):
        video_name = example["video_name"]
        video_path = os.path.join(video_dir, split, f"{video_name}.mp4")

        if not os.path.exists(video_path):
            print(f"Video non trovato: {video_path}, skipping.")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Impossibile aprire il video: {video_path}, skipping.")
            continue

        frame_landmarks = []
        frame_id = 0

        # Estrazione dei frame dal video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Converte il frame da BGR (usato da OpenCV) a RGB (usato da MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = [
                        {"x": landmark.x, "y": landmark.y, "z": landmark.z}
                        for landmark in face_landmarks.landmark
                    ]
                    frame_landmarks.append(
                        {"frame_id": frame_id + 1, "landmarks": landmarks}
                    )
            frame_id += 1

        cap.release()

        # Salva i landmark come JSON per ogni video
        if frame_landmarks:
            output_filename = f"{video_name}_landmarks.json"
            output_path = os.path.join(split_output_dir, output_filename)
            with open(output_path, "w") as f:
                json.dump(frame_landmarks, f)

print("Estrazione completata!")
