# PANORAMICA DEL FLUSSO:
# Questo script elabora i file video grezzi per estrarre i landmark facciali
# utilizzando la libreria MediaPipe. Per ogni frame campionato da un video,
# rileva i punti chiave del viso e li salva in un file JSON.
# L'output è organizzato in una struttura di cartelle che rispecchia quella
# attesa dalla pipeline di dati successiva (una cartella per video, un JSON per frame).

import cv2
import mediapipe as mp
import os
import json
import logging
from tqdm import tqdm

# --- Sezione 1: Configurazione Iniziale ---

# Configura il logging per mostrare messaggi informativi durante l'esecuzione.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Costruisce percorsi assoluti per le directory di input e output.
# Questo rende lo script eseguibile da qualsiasi posizione, senza dipendere dalla
# directory di lavoro corrente.
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
VIDEO_DIR = os.path.join(BASE_DIR, "data", "raw", "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "interim", "mediapipe_landmarks")

# Crea la directory di output se non esiste.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Inizializza il modello Face Mesh di MediaPipe.
# - static_image_mode=False: Ottimizzato per i video.
# - max_num_faces=1: Cerca un solo volto per frame, per efficienza.
# - min_detection_confidence=0.5: Soglia di confidenza per il rilevamento.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5
)


# --- Sezione 2: Funzione di Estrazione dei Landmark ---
def process_video(video_path, video_output_dir):
    """
    Elabora un singolo video, estrae i landmark facciali e li salva in file JSON.

    Args:
        video_path (str): Percorso del file video da elaborare.
        video_output_dir (str): Directory in cui salvare i file JSON per ogni frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.warning(f"Impossibile aprire il video: {video_path}. Verrà saltato.")
        return

    # --- Campionamento dei Frame ---
    # Per efficienza, campioniamo i video a un framerate target (es. 10 FPS).
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 10
    frame_skip = (
        round(original_fps / target_fps) if original_fps > 0 and target_fps > 0 else 1
    )
    if frame_skip < 1:
        frame_skip = 1

    frame_count = 0
    processed_frames = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Processa solo i frame che corrispondono al nostro intervallo di campionamento.
        if frame_count % frame_skip == 0:
            # Converte il frame da BGR (usato da OpenCV) a RGB (usato da MediaPipe).
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            # Se viene rilevato un volto, estrae e salva i landmark.
            if results.multi_face_landmarks:
                # Prende i landmark del primo (e unico) volto rilevato.
                face_landmarks = results.multi_face_landmarks[0]

                # Estrae le coordinate (x, y, z) e le appiattisce in una lista.
                # Questo formato è simile a quello di OpenPose (senza punteggi di confidenza).
                flat_landmarks = []
                for landmark in face_landmarks.landmark:
                    flat_landmarks.extend([landmark.x, landmark.y, landmark.z])

                # Crea una struttura dati simile a quella di OpenPose per compatibilità.
                # La chiave "face_keypoints_2d" è usata per distinguerli da "pose_keypoints_2d".
                output_data = {
                    "version": 1.3,
                    "people": [
                        {
                            "face_keypoints_2d": flat_landmarks,
                            "pose_keypoints_2d": [],  # Lasciato vuoto se non si usa OpenPose
                        }
                    ],
                }

                # Salva i dati del frame in un file JSON.
                # Il nome del file mantiene l'ordine cronologico.
                output_filename = os.path.join(
                    video_output_dir, f"{processed_frames:012d}_keypoints.json"
                )
                with open(output_filename, "w") as f:
                    json.dump(output_data, f)

                processed_frames += 1

        frame_count += 1

    cap.release()
    if processed_frames > 0:
        logging.info(
            f"Salvati {processed_frames} frame di landmark per il video: {os.path.basename(video_path)}"
        )
    else:
        logging.warning(
            f"Nessun volto rilevato nel video: {os.path.basename(video_path)}"
        )


# --- Sezione 3: Ciclo Principale di Elaborazione ---
def main():
    """
    Scansiona la directory dei video ed esegue l'estrazione dei landmark per ciascuno.
    """
    video_files = [
        f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".avi", ".mov"))
    ]
    if not video_files:
        logging.warning(f"Nessun file video trovato in {VIDEO_DIR}. Uscita.")
        return

    logging.info(f"Trovati {len(video_files)} video da elaborare.")

    for video_file in tqdm(video_files, desc="Elaborazione video"):
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(VIDEO_DIR, video_file)

        # Crea una directory specifica per i landmark di questo video.
        video_output_dir = os.path.join(OUTPUT_DIR, video_name)

        # Controlla se i dati per il video sono già stati estratti.
        if os.path.exists(video_output_dir) and os.listdir(video_output_dir):
            logging.info(f"Landmark già estratti per {video_file}. Saltato.")
            continue

        os.makedirs(video_output_dir, exist_ok=True)
        process_video(video_path, video_output_dir)

    logging.info("Estrazione completata per tutti i video!")


if __name__ == "__main__":
    main()
