# =================================================================================================
# COMANDI UTILI PER L'ESECUZIONE
# =================================================================================================
#
# 1. AVVIARE IL SERVER MLFLOW
#    Per monitorare gli esperimenti, prima di eseguire lo script di tuning,
#    aprire un terminale e lanciare il server MLflow:
#
#    mlflow server --host 127.0.0.1 --port 8080
#
# 2. ESEGUIRE L'OTTIMIZZAZIONE DEGLI IPERPARAMETRI
#    Questo script utilizza Optuna per trovare i migliori iperparametri per un dato modello.
#    È possibile specificare il tipo di modello (lstm o stgcn), il numero di "trial" (tentativi)
#    e il numero di epoche per ogni trial.
#
#    ESEMPIO DI COMANDO PER LSTM:
#    Esegue 40 tentativi di ottimizzazione per il modello LSTM, addestrando per 40 epoche ogni volta.
#
#    python3 src/models/hyperparameter_tuning.py --model_type lstm --n_trials 40 --num_epochs 40
#
#    ESEMPIO DI COMANDO PER ST-GCN:
#    Esegue 20 tentativi di ottimizzazione per il modello ST-GCN, addestrando per 25 epoche ogni volta.
#
#    python3 src/models/hyperparameter_tuning.py --model_type stgcn --n_trials 20 --num_epochs 25
#
# =================================================================================================

import os
import sys
import argparse
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import f1_score
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from optuna.pruners import MedianPruner
import random
from torch.backends import cudnn
from optuna.samplers import TPESampler

# Ignite imports
# Workaround for MPS OOM: disable MPS upper memory limit (may risk system stability)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy
from ignite.handlers import EarlyStopping as IgniteEarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from optuna.integration import PyTorchIgnitePruningHandler
from optuna.exceptions import TrialPruned


# --- Sezione 1: Setup del Percorso di Base ---
# Aggiunge la directory radice del progetto (src) al path di Python per importare moduli custom.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

from data_pipeline.landmark_dataset import LandmarkDataset
from models.lstm_model import EmotionLSTM
from models.stgcn_model import STGCN

# --- Sezione 2: Setup di MLflow ---
# Imposta l'URI del server MLflow e il nome dell'esperimento.
# Se l'esperimento non esiste, viene creato.
mlflow.set_tracking_uri("http://127.0.0.1:8080")
experiment_name = "Emotion Recognition Experiment"
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name)
else:
    mlflow.set_experiment(experiment_name)

# --- Sezione 3: Definizione dei Percorsi Dati ---
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

# --- Sezione 4: Variabili Globali ---
# Variabili globali che verranno impostate dalla linea di comando.
NUM_EPOCHS = 5
MODEL_TYPE = "lstm"
PATIENCE = 10  # Pazienza per l'early stopping


def prepare_batch(batch, device, non_blocking=False):
    """
    Prepara un batch di dati per l'addestramento.
    Sposta i tensori sul dispositivo corretto e rimodella l'input per ST-GCN.
    """
    x, y = batch
    if MODEL_TYPE == "stgcn":
        # Rimodella l'input da (B, T, F) a (B, T, N, C)
        # B=batch, T=time, F=features, N=numero di punti, C=canali (coordinate)
        b, t, f = x.shape
        coords_per_point = 2
        num_point = f // coords_per_point
        x = x.view(b, t, num_point, coords_per_point)
    return (
        x.to(device, non_blocking=non_blocking),
        y.to(device, non_blocking=non_blocking),
    )


def objective(trial):
    """
    Funzione "obiettivo" di Optuna.
    Questa funzione definisce, esegue e valuta un singolo "trial" (tentativo) di
    ottimizzazione degli iperparametri.
    """
    # --- Sezione 5: Definizione dello Spazio di Ricerca degli Iperparametri ---
    # Optuna suggerisce i valori per gli iperparametri da uno spazio di ricerca definito.
    # Lo spazio di ricerca è diverso a seconda del tipo di modello.
    if MODEL_TYPE == "lstm":
        # Spazio di ricerca per LSTM
        hidden_size = trial.suggest_int("hidden_size", 768, 1024, step=32)
        num_layers = trial.suggest_int("num_layers", 2, 5)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)
        batch_size = trial.suggest_categorical("batch_size", [64, 96, 128, 160])
        params_to_log = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
        }
    else:  # stgcn
        # Spazio di ricerca per ST-GCN
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        dropout = trial.suggest_float(
            "dropout", 0, 0.2
        )  # NOTA: testare anche senza dropout
        params_to_log = {"stgcn_dropout": dropout}

    # --- Sezione 6: Caricamento Dati e Gestione Sbilanciamento ---
    train_dataset = LandmarkDataset(
        landmarks_dir=TRAIN_LANDMARKS_DIR, processed_file=TRAIN_PROCESSED_FILE
    )
    val_dataset = LandmarkDataset(
        landmarks_dir=VAL_LANDMARKS_DIR, processed_file=VAL_PROCESSED_FILE
    )

    # --- Weighted Sampler per gestire lo sbilanciamento delle classi ---
    # Crea un sampler che bilancia le classi in ogni batch.
    labels_arr = train_dataset.processed["emotion"].map(train_dataset.label_map).values
    class_counts = np.bincount(labels_arr)
    class_weights_for_sampler = 1.0 / class_counts
    sample_weights = np.array(
        [class_weights_for_sampler[label] for label in labels_arr]
    )
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # I DataLoader utilizzano il sampler per il training.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Pesi per la Funzione di Loss ---
    # Calcola i pesi per la loss function per penalizzare maggiormente gli errori sulle classi minoritarie.
    labels_arr = train_dataset.processed["emotion"].map(train_dataset.label_map).values
    class_counts = np.bincount(labels_arr)
    num_classes = len(class_counts)
    total_samples = len(labels_arr)
    class_weights = torch.tensor(
        [total_samples / (num_classes * count) for count in class_counts],
        dtype=torch.float32,
    )

    # --- Sezione 7: Setup del Modello e dell'Addestramento ---
    # Seleziona il dispositivo (CPU, GPU, MPS).
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    class_weights = class_weights.to(device)

    # Istanzia il modello corretto (LSTM o ST-GCN) con gli iperparametri suggeriti da Optuna.
    input_size = train_dataset[0][0].shape[1]
    num_classes = len(train_dataset.labels)
    if MODEL_TYPE == "lstm":
        model = EmotionLSTM(
            input_size, hidden_size, num_layers, num_classes, dropout
        ).to(device)
    else:
        coords_per_point = 2
        num_point = input_size // coords_per_point
        model = STGCN(
            num_classes,
            num_point,
            num_person=1,
            in_channels=coords_per_point,
            dropout_rate=dropout,
        ).to(device)

    # Inizializza loss, ottimizzatore e scheduler.
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=3
    )  # Monitora val_f1_macro

    # Crea trainer e evaluator di Ignite.
    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device, prepare_batch=prepare_batch
    )
    evaluator = create_supervised_evaluator(
        model,
        metrics={"val_loss": Loss(criterion), "val_acc": Accuracy()},
        device=device,
        prepare_batch=prepare_batch,
    )

    # Funzione per calcolare e aggiungere gli F1 score alle metriche alla fine della validazione.
    @evaluator.on(Events.COMPLETED)
    def compute_f1_scores(engine):
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                # Il reshaping è gestito da prepare_batch
                xb, yb = prepare_batch((xb, yb), device)
                outputs = model(xb.float())
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        engine.state.metrics["val_f1_macro"] = f1_macro
        engine.state.metrics["val_f1"] = f1_weighted
        engine.state.metrics["val_f1_class_0"] = float(f1_per_class[0])
        engine.state.metrics["val_f1_class_1"] = (
            float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0
        )

    # --- Sezione 8: Esecuzione del Trial e Logging ---
    # Avvia una run nidificata in MLflow per questo specifico trial.
    with mlflow.start_run(nested=True):
        # Logga i parametri comuni e quelli specifici del modello.
        base_params = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "model_type": MODEL_TYPE,
        }
        mlflow.log_params({**base_params, **params_to_log})

        # Logga la mappatura delle classi (es. 0: 'neutral', 1: 'emotion').
        class_map = {i: label for i, label in enumerate(train_dataset.labels)}
        mlflow.log_params({f"class_{i}": label for i, label in class_map.items()})

        best_metrics = {"val_loss": float("inf"), "val_f1": 0.0, "val_f1_macro": 0.0}

        # Funzione eseguita alla fine di ogni epoca di addestramento.
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            # Calcola la loss media sul training set.
            model.eval()
            train_loss_sum = 0.0
            with torch.no_grad():
                for xb, yb in train_loader:
                    # Il reshaping è gestito da prepare_batch
                    xb, yb = prepare_batch((xb, yb), device)
                    outputs = model(xb.float())
                    loss = criterion(outputs, yb)
                    train_loss_sum += loss.item() * xb.size(0)
            train_loss = train_loss_sum / len(train_loader.dataset)

            # Esegue la validazione e ottiene le metriche.
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            val_loss = metrics["val_loss"]
            val_acc = metrics["val_acc"]
            val_f1 = metrics["val_f1"]
            val_f1_macro = metrics["val_f1_macro"]

            # Aggiorna le migliori metriche del trial.
            if val_f1_macro > best_metrics["val_f1_macro"]:
                best_metrics["val_f1_macro"] = val_f1_macro

            if val_loss < best_metrics["val_loss"]:
                best_metrics["val_loss"] = val_loss
            if val_f1 > best_metrics["val_f1"]:
                best_metrics["val_f1"] = val_f1

            # Logga le metriche dell'epoca su MLflow.
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
            # Aggiorna lo scheduler e svuota la cache.
            scheduler.step(
                val_f1_macro
            )  # Monitora val_f1_macro per gestire lo sbilanciamento
            # Clear GPU/MPS cache to prevent OOM
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

        # --- Sezione 9: Pruning e Early Stopping ---
        # Pruning: meccanismo di Optuna per interrompere i trial poco promettenti.
        # Monitora 'val_f1_macro' e ferma il trial se le performance sono scarse.
        pruning_handler = PyTorchIgnitePruningHandler(trial, "val_f1_macro", trainer)
        evaluator.add_event_handler(Events.COMPLETED, pruning_handler)

        # Early Stopping: ferma l'addestramento se 'val_f1_macro' non migliora per 'PATIENCE' epoche.
        early_stopper = IgniteEarlyStopping(
            patience=PATIENCE,
            score_function=lambda eng: eng.state.metrics["val_f1_macro"],
            trainer=trainer,
        )
        evaluator.add_event_handler(Events.COMPLETED, early_stopper)

        try:
            # Avvia l'addestramento.
            trainer.run(train_loader, max_epochs=NUM_EPOCHS)
        except TrialPruned:
            # Se il trial viene "potato" (pruned), lo segnala in MLflow e solleva l'eccezione.
            mlflow.set_tag("status", "pruned")
            raise

        # Logga le migliori metriche ottenute alla fine del trial.
        mlflow.log_metrics(
            {
                "best_val_loss": best_metrics["val_loss"],
                "best_val_f1": best_metrics["val_f1"],
                "best_val_f1_macro": best_metrics["val_f1_macro"],
            }
        )
        # Svuota la cache alla fine del trial.
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        # Restituisce la metrica che Optuna deve ottimizzare (massimizzare).
        return best_metrics["val_f1_macro"]


def main():
    """
    Funzione principale che orchestra l'intero processo di ottimizzazione.
    """
    # --- Sezione 10: Parsing degli Argomenti e Setup dello Studio ---
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning with Optuna and MLflow"
    )
    parser.add_argument(
        "--model_type", type=str, choices=["lstm", "stgcn"], default="lstm"
    )
    parser.add_argument(
        "--n_trials", type=int, default=20, help="Number of Optuna trials"
    )
    parser.add_argument("--num_epochs", type=int, default=5, help="Epochs per trial")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    # Imposta i seed per la riproducibilità.
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Imposta le variabili globali con i valori passati da linea di comando.
    global NUM_EPOCHS, MODEL_TYPE
    NUM_EPOCHS = args.num_epochs
    MODEL_TYPE = args.model_type
    # Avvia una run "genitore" in MLflow per l'intero studio di ottimizzazione.
    run_name = f"Optuna_{MODEL_TYPE}_tuning"
    with mlflow.start_run(run_name=run_name):
        # Logga il seed per la riproducibilità.
        mlflow.log_param("seed", seed)
        # Crea lo "studio" di Optuna, specificando la direzione (massimizzare f1_macro),
        # il pruner e il sampler.
        study = optuna.create_study(
            direction="maximize",  # Massimizza f1_macro
            pruner=MedianPruner(n_startup_trials=PATIENCE, n_warmup_steps=1),
            sampler=TPESampler(seed=seed),
        )
        # Avvia l'ottimizzazione.
        study.optimize(objective, n_trials=args.n_trials)
        # Alla fine dello studio, logga le informazioni globali su MLflow.
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_val_f1_macro_study", study.best_value)
        mlflow.set_tags(
            {
                "project": "EmoSign",
                "optimizer_engine": "optuna",
                "model_family": MODEL_TYPE,
            }
        )
    print("Best trial:", study.best_trial.params)


if __name__ == "__main__":
    main()
