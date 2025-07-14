# =================================================================================================
# COMANDI UTILI PER L'ESECUZIONE
# =================================================================================================
#
# 1. AVVIARE IL SERVER MLFLOW
#    Per monitorare gli esperimenti, prima di eseguire lo script di training,
#    aprire un terminale e lanciare il server MLflow:
#
#    mlflow server --host 127.0.0.1 --port 8080
#
# 2. ESEGUIRE L'ADDESTRAMENTO DEL MODELLO
#    Questo script permette di addestrare un modello per il riconoscimento delle emozioni.
#    È possibile specificare il tipo di modello (lstm o stgcn) e i relativi iperparametri.
#
#    ESEMPIO DI COMANDO PER LSTM (con iperparametri ottimizzati):
#    Questo comando avvia l'addestramento di un modello LSTM con i migliori iperparametri
#    trovati durante la fase di tuning.
#
#    python src/models/run_train.py \
#      --model_type lstm \
#      --batch_size 128 \
#      --hidden_size 896 \
#      --num_layers 3 \
#      --learning_rate 2.621087878265438e-05 \
#      --dropout 0.23993475643167195 \
#      --num_epochs 100 \
#      --seed 44
#
#    ESEMPIO DI COMANDO PER ST-GCN (da personalizzare):
#    Questo comando avvia l'addestramento di un modello ST-GCN.
#    Gli iperparametri come learning_rate e dropout andrebbero ottimizzati.
#
#    python src/models/run_train.py \
#      --model_type stgcn \
#      --batch_size 64 \
#      --learning_rate 1e-4 \
#      --dropout 0.1 \
#      --num_epochs 100 \
#      --seed 44
#
# =================================================================================================


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
import numpy as np
import os, sys
import argparse
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
import requests  # for pinging MLflow server

# --- Sezione 1: Setup del Percorso di Base ---
# Aggiunge la directory radice del progetto (src) al path di Python.
# Questo permette di importare moduli custom come LandmarkDataset e i modelli.
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
sys.path.insert(0, BASE_DIR)

# Ignite imports
# Workaround for MPS OOM: disable MPS upper memory limit (may risk system stability)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy
from ignite.handlers import EarlyStopping as IgniteEarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data_pipeline.landmark_dataset import (
    LandmarkDataset,
)  # Importa la classe per gestire i dati
from src.models.lstm_model import EmotionLSTM  # Importa l'architettura del modello LSTM
from src.models.stgcn_model import STGCN  # Aggiunto per ST-GCN support
from torch.backends import cudnn
import random


def prepare_batch(batch, device, model_type, non_blocking=False):
    """
    Prepara un batch di dati per l'addestramento.
    Sposta i tensori sul dispositivo corretto (CPU, GPU, MPS) e, se il modello è ST-GCN,
    rimodella i dati di input nella forma attesa (B, T, N, C).
    """
    x, y = batch
    if model_type == "stgcn":
        # Reshape input from (B, T, F) to (B, T, N, C)
        # B=batch, T=time, F=features, N=num_points, C=channels
        b, t, f = x.shape
        coords_per_point = 2
        num_point = f // coords_per_point
        x = x.view(b, t, num_point, coords_per_point)
    return (
        x.to(device, non_blocking=non_blocking),
        y.to(device, non_blocking=non_blocking),
    )


def main(args):
    """Funzione principale che orchestra l'addestramento e la valutazione del modello."""
    # --- Sezione 2: Setup dell'Ambiente ---
    # Imposta i seed per la riproducibilità degli esperimenti.
    # In questo modo, eseguendo lo script con gli stessi parametri si otterranno gli stessi risultati.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # Seleziona il dispositivo (device) per l'addestramento.
    # Dà priorità a MPS (per Mac con Apple Silicon), poi a CUDA (per GPU NVIDIA), e infine alla CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training su dispositivo: {device}")

    # Imposta l'URI del server MLflow per il tracciamento degli esperimenti.
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    # Imposta (o crea se non esiste) l'esperimento MLflow in cui verranno loggati i dati.
    mlflow.set_experiment("Emotion Recognition Experiment")

    # --- Sezione 3: Caricamento Dati e Creazione del Modello ---
    # Definisce i percorsi per i dati di training e validazione.
    TRAIN_LANDMARKS_DIR = os.path.join(
        BASE_DIR, "data", "raw", "train", "openpose_output_train", "json"
    )
    TRAIN_PROCESSED_FILE = os.path.join(
        BASE_DIR, "data", "processed", "train", "video_sentiment_data_0.65.csv"
    )
    VAL_LANDMARKS_DIR = os.path.join(
        BASE_DIR, "data", "raw", "val", "openpose_output_val", "json"
    )
    VAL_PROCESSED_FILE = os.path.join(
        BASE_DIR, "data", "processed", "val", "video_sentiment_data_0.65.csv"
    )

    # Crea le istanze del dataset per training e validazione.
    train_dataset = LandmarkDataset(
        landmarks_dir=TRAIN_LANDMARKS_DIR, processed_file=TRAIN_PROCESSED_FILE
    )
    val_dataset = LandmarkDataset(
        landmarks_dir=VAL_LANDMARKS_DIR, processed_file=VAL_PROCESSED_FILE
    )

    # --- Gestione dello Sbilanciamento delle Classi (Weighted Sampler) ---
    # Per contrastare lo sbilanciamento delle classi nel dataset, viene usato un WeightedRandomSampler.
    # Questo sampler assicura che ogni batch contenga una rappresentazione bilanciata delle classi,
    # campionando più frequentemente gli esempi delle classi meno numerose.
    labels_arr_sampler = (
        train_dataset.processed["emotion"].map(train_dataset.label_map).values
    )
    class_counts_sampler = np.bincount(labels_arr_sampler)
    class_weights_for_sampler = 1.0 / class_counts_sampler
    sample_weights = np.array(
        [class_weights_for_sampler[label] for label in labels_arr_sampler]
    )
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Crea i DataLoader. Per il training, si usa il sampler (shuffle=False è obbligatorio con sampler).
    # Per la validazione, non si usa il sampler per valutare le performance sulla distribuzione reale dei dati.
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler
    )
    # The validation loader remains unchanged to evaluate on the real, imbalanced data distribution.
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Pesi per la Funzione di Loss ---
    # Come ulteriore meccanismo per gestire lo sbilanciamento, si calcolano dei pesi per la CrossEntropyLoss.
    # Questi pesi assegnano una penalità maggiore agli errori sulle classi minoritarie.
    labels_array = (
        train_dataset.processed["emotion"].map(train_dataset.label_map).values
    )
    class_counts = np.bincount(labels_array)
    class_weights = [len(labels_array) / (len(class_counts) * c) for c in class_counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Determina la dimensione di input e il numero di classi, poi crea il modello.
    # La scelta tra LSTM e ST-GCN dipende dall'argomento --model_type.
    input_size = train_dataset[0][0].shape[1]
    num_classes = len(train_dataset.labels)
    if args.model_type == "lstm":
        model = EmotionLSTM(
            input_size,
            args.hidden_size,
            args.num_layers,
            num_classes,
            dropout=args.dropout,
        ).to(device)
    elif args.model_type == "stgcn":
        coords_per_point = 2
        num_point = input_size // coords_per_point
        model = STGCN(
            num_classes,
            num_point,
            num_person=1,
            in_channels=coords_per_point,
            dropout_rate=args.dropout,
        ).to(device)

    # Inizializza la funzione di loss, l'ottimizzatore e lo scheduler per il learning rate.
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # Lo scheduler riduce il learning rate se la metrica monitorata (val_f1_macro) non migliora.
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=3
    )  # NOTE New: monitors val_f1_macro

    # Crea una funzione `prepare_batch_fn` usando una lambda per passare `args.model_type`.
    # Questo è necessario perché le funzioni di Ignite accettano solo `batch, device, non_blocking`.
    prepare_batch_fn = lambda batch, device, non_blocking: prepare_batch(
        batch, device, args.model_type, non_blocking
    )

    # Crea il trainer di Ignite, che gestisce il ciclo di addestramento.
    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device, prepare_batch=prepare_batch_fn
    )

    # --- Sezione 4: Definizione delle Metriche e degli Handlers di Ignite ---
    # Definisce le metriche di base (loss e accuracy) per il set di validazione.
    val_metrics = {
        "val_loss": Loss(criterion, device="cpu"),
        "val_acc": Accuracy(device="cpu"),
    }

    # Crea l'evaluator di Ignite, che gestisce il ciclo di valutazione.
    evaluator = create_supervised_evaluator(
        model, metrics=val_metrics, device=device, prepare_batch=prepare_batch_fn
    )

    # Aggiunge un evento per calcolare gli F1 score alla fine di ogni epoca di validazione.
    # L'F1 score (specialmente macro) è una metrica cruciale per problemi sbilanciati.
    @evaluator.on(Events.COMPLETED)
    def compute_f1_scores(engine):
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                # Reshaping is now handled by prepare_batch
                xb, yb = prepare_batch_fn((xb, yb), device, non_blocking=False)
                outputs = model(xb.float())
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # Calcola le diverse versioni dell'F1 score.
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Aggiunge le metriche calcolate allo stato dell'evaluator per il logging.
        engine.state.metrics["val_f1_macro"] = f1_macro
        engine.state.metrics["val_f1"] = f1_weighted
        engine.state.metrics["val_f1_class_0"] = float(f1_per_class[0])
        engine.state.metrics["val_f1_class_1"] = (
            float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0
        )

    # --- Handlers di Ignite ---
    # ModelCheckpoint: Salva il modello migliore basandosi sul 'val_f1_macro'.
    # Verrà salvato solo il modello che ottiene il punteggio F1 macro più alto sulla validazione.
    checkpoint_handler = ModelCheckpoint(
        dirname=os.path.join(BASE_DIR, "models"),
        filename_prefix=f"emotion_{args.model_type}",
        n_saved=1,
        create_dir=True,
        require_empty=False,
        score_function=lambda eng: eng.state.metrics["val_f1_macro"],
        score_name="val_f1_macro",
    )
    evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler, {"model": model})

    # EarlyStopping: Ferma l'addestramento se il 'val_f1_macro' non migliora per un numero
    # definito di epoche ('patience'), evitando overfitting.
    earlystop_handler = IgniteEarlyStopping(
        patience=args.patience,
        score_function=lambda eng: eng.state.metrics["val_f1_macro"],
        trainer=trainer,
    )
    evaluator.add_event_handler(Events.COMPLETED, earlystop_handler)

    # --- Sezione 5: Ciclo di Addestramento e Logging con MLflow ---
    print("Inizio training con validazione (Ignite)...")
    run_name = f"EmotionRecognition_{args.model_type}_train"

    # Avvia una run di MLflow per tracciare parametri e metriche di questo specifico addestramento.
    with mlflow.start_run(run_name=run_name) as run:
        # Logga i parametri utilizzati per questo addestramento.
        params = {
            "model_type": args.model_type,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "seed": args.seed,
        }
        if args.model_type == "lstm":
            params["hidden_size"] = args.hidden_size
            params["num_layers"] = args.num_layers
            params["dropout"] = args.dropout
        else:  # stgcn
            params["stgcn_dropout"] = args.dropout

        mlflow.log_params(params)
        # Dizionario per tenere traccia delle migliori metriche ottenute.
        best_metrics = {"val_loss": float("inf"), "val_f1": 0.0, "val_f1_macro": 0.0}

        # Definisce una funzione da eseguire alla fine di ogni epoca del trainer.
        @trainer.on(Events.EPOCH_COMPLETED)
        def validate_and_log(engine):
            # Calcola la loss media sul training set per l'epoca corrente.
            model.eval()
            train_loss_sum = 0.0
            with torch.no_grad():
                for xb, yb in train_loader:
                    # Reshaping is now handled by prepare_batch
                    xb, yb = prepare_batch_fn((xb, yb), device, non_blocking=False)
                    outputs = model(xb.float())
                    loss = criterion(outputs, yb)
                    train_loss_sum += loss.item() * xb.size(0)
            train_loss = train_loss_sum / len(train_loader.dataset)

            # Esegue il ciclo di validazione per calcolare le metriche.
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            val_loss = metrics["val_loss"]
            val_acc = metrics["val_acc"]
            val_f1 = metrics["val_f1"]
            val_f1_macro = metrics["val_f1_macro"]

            # Aggiorna le migliori metriche se quelle correnti sono migliori.
            if val_loss < best_metrics["val_loss"]:
                best_metrics["val_loss"] = val_loss
            if val_f1 > best_metrics["val_f1"]:
                best_metrics["val_f1"] = val_f1
            if val_f1_macro > best_metrics["val_f1_macro"]:
                best_metrics["val_f1_macro"] = val_f1_macro

            # Logga le metriche dell'epoca corrente su MLflow.
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "val_f1_macro": val_f1_macro,
                    "val_f1_class_0": metrics["val_f1_class_0"],
                    "val_f1_class_1": metrics["val_f1_class_1"],
                },
                step=engine.state.epoch,
            )
            # Stampa a console un riepilogo dell'epoca.
            print(
                f"Epoch {engine.state.epoch}/{args.num_epochs}, "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val F1 Macro: {val_f1_macro:.4f}"
            )
            # Aggiorna lo scheduler del learning rate.
            scheduler.step(
                val_f1_macro
            )  # NOTE New: monitors val_f1_macro to better handle imbalance

            # Svuota la cache della GPU/MPS per liberare memoria.
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

        # Avvia il ciclo di addestramento.
        trainer.run(train_loader, max_epochs=args.num_epochs)

        print(
            f"Training completato. Best model saved in {checkpoint_handler.last_checkpoint}"
        )

        # Alla fine del training, logga le migliori metriche ottenute in assoluto.
        mlflow.log_metric("best_val_loss", best_metrics["val_loss"])
        mlflow.log_metric("best_val_f1", best_metrics["val_f1"])
        mlflow.log_metric("best_val_f1_macro", best_metrics["val_f1_macro"])

        # --- Log del Modello su MLflow ---
        # Inferisce la "firma" del modello (input e output) e lo salva su MLflow.
        # Questo permette di riutilizzare facilmente il modello in futuro.
        data_sample, _ = next(iter(train_loader))
        # Reshape sample for ST-GCN if necessary
        data_sample, _ = prepare_batch_fn((data_sample, _), device, non_blocking=False)

        signature = infer_signature(
            data_sample.cpu().numpy(),
            model(data_sample.float()).cpu().detach().numpy(),
        )
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="EmoSign_pytorch",
        )
        mlflow.set_tag("experiment_purpose", "PyTorch emotion recognition")


# --- Sezione 6: Parsing degli Argomenti da Linea di Comando ---
# Definisce gli argomenti che possono essere passati allo script da terminale,
# con valori di default e descrizioni.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Emotion Recognition Model")
    parser.add_argument(
        "--model_type", type=str, default="lstm", choices=["lstm", "stgcn"]
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    main(args)

# python train_model.py --batch_size 64 --hidden_size 512 --num_layers 3 --learning_rate 0.0005 --dropout 0.3
