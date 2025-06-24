# ESEMPIO DI OUTPUT JSON

# [
#   {
#     "frame_id": 1,
#     "landmarks": [
#       {"x": 0.123, "y": 0.456, "z": 0.789},
#       {"x": 0.223, "y": 0.356, "z": 0.689},
#       ...
#     ]
#   },
#   {
#     "frame_id": 2,
#     "landmarks": [
#       {"x": 0.128, "y": 0.412, "z": 0.710},
#       ...
#     ]
#   }
# ]


import cv2
import mediapipe as mp
import os
import json
import logging
from tqdm import tqdm

# Configura il logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Percorsi delle directory
VIDEO_DIR = "../../data/raw/videos/"
OUTPUT_DIR = "../../data/interim/landmarks/"

# Creazione directory per landmark estratti
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Inizializza il rilevamento di Face Mesh con Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5
)


# Estrattore di landmark
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_landmarks = []

    if not cap.isOpened():
        logging.warning(f"Impossibile aprire il video: {video_path}. Verrà saltato.")
        return

    # --- Campionamento dei Frame ---
    # Per allinearsi con il paper EmoSign e per efficienza, campioniamo i video
    # a un framerate target (es. 10 FPS) "al volo" durante l'elaborazione.
    # Questo approccio evita di creare copie modificate dei video, conservando
    # i dati grezzi originali e mantenendo la flessibilità del processo.
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 10

    # Calcola quanti frame saltare per raggiungere il target_fps.
    # Se l'FPS originale è 30 e il target è 10, processeremo 1 frame ogni 3 (30/10=3).
    frame_skip = round(original_fps / target_fps) if original_fps > 0 else 1
    if frame_skip < 1:
        frame_skip = 1

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Processa solo i frame che corrispondono al nostro intervallo di campionamento.
        if frame_count % frame_skip == 0:
            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Conversione del frame in RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Elaborazione del frame per rilevamento landmark
            results = face_mesh.process(frame_rgb)

            # Se il viso è rilevato, salviamo i landmark
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = [
                        {
                            "x": landmark.x,  # Normalizzato da 0 a 1
                            "y": landmark.y,  # Normalizzato da 0 a 1
                            "z": landmark.z,  # Normalizzato rispetto alla dimensione
                        }
                        for landmark in face_landmarks.landmark
                    ]

                    frame_landmarks.append(
                        {"frame_id": frame_id, "landmarks": landmarks}
                    )

        frame_count += 1

    cap.release()

    # Scrive i dati JSON per ogni video
    with open(output_path, "w") as f:
        json.dump(frame_landmarks, f)

    logging.info(f"Landmark salvati per il video: {os.path.basename(video_path)}")


# Elaborazione dei video nella cartella VIDEO_DIR
for video_file in tqdm(os.listdir(VIDEO_DIR)):
    video_path = os.path.join(VIDEO_DIR, video_file)
    output_path = os.path.join(
        OUTPUT_DIR, f"{os.path.splitext(video_file)[0]}_landmarks.json"
    )

    # Controlla se i dati per il video sono già stati calcolati
    if os.path.exists(output_path):
        logging.info(f"Landmark già estratti per {video_file}. Saltato.")
        continue

    # Processa il video
    process_video(video_path, output_path)

logging.info("Estratto completato per tutti i video!")
