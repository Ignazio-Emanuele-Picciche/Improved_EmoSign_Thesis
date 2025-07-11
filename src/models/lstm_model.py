# PANORAMICA DEL FLUSSO:
# Questo file definisce l'architettura della nostra rete neurale, il "cervello" che
# imparerà a classificare le emozioni.
# La classe `EmotionLSTM` è un modello basato su Long Short-Term Memory (LSTM),
# una tipologia di rete particolarmente adatta a elaborare dati sequenziali come
# le nostre serie temporali di landmark facciali.
# Il suo compito è ricevere una sequenza di landmark e produrre in output le probabilità
# per ciascuna classe di emozione.

# ARCHITETTURA DEL MODELLO:
# La rete è composta da un layer LSTM che elabora la sequenza temporale e
# un layer fully connected (lineare) che mappa l'output dell'LSTM sul numero di classi di emozioni.
# Il layer LSTM è configurato con un certo numero di neuroni (hidden_size) e
# può essere composto da più strati (num_layers).
# Il dropout è applicato tra i layer per prevenire l'overfitting, "spegnendo" casualmente
# alcuni neuroni durante l'addestramento.

import torch
import torch.nn as nn


class EmotionLSTM(nn.Module):
    # Il metodo __init__ definisce i "mattoni" della nostra rete neurale.
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        """
        Args:
            input_size (int): Dimensione del vettore di feature per ogni frame (468 * 3 = 1404).
            hidden_size (int): Numero di neuroni nel layer LSTM. Un valore più alto aumenta la capacità del modello.
            num_layers (int): Numero di layer LSTM da impilare. Più layer permettono di imparare pattern più complessi.
            num_classes (int): Numero di classi di output, cioè il numero di emozioni da riconoscere (es. 7).
            dropout (float): Tecnica di regolarizzazione per prevenire l'overfitting, "spegnendo" casualmente alcuni neuroni.
        """
        super(EmotionLSTM, self).__init__()
        # 1. Il layer LSTM: è il cuore del modello, processa la sequenza temporale.
        # `batch_first=True` significa che i dati in input avranno la dimensione del batch come prima dimensione.
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        # 2. Il layer Fully Connected (o Lineare): prende l'output finale dell'LSTM e lo mappa
        # sul numero di classi di emozioni desiderate.
        self.fc = nn.Linear(hidden_size, num_classes)

    # Il metodo `forward` definisce come i dati fluiscono attraverso i layer definiti in __init__.
    def forward(self, x):
        # `x` è il tensore di input con forma (batch_size, seq_length, input_size)

        # 1. Passa la sequenza di input attraverso l'LSTM.
        # L'LSTM restituisce due cose: `output` (lo stato nascosto per ogni timestep) e
        # `(h_n, c_n)` (l'ultimo stato nascosto e l'ultimo stato della cella).
        # A noi interessa solo l'ultimo stato nascosto `h_n`, che riassume l'intera sequenza.
        _, (h_n, _) = self.lstm(x)

        # 2. `h_n` ha forma (num_layers, batch_size, hidden_size).
        # Prendiamo lo stato nascosto dell'ultimo layer LSTM (`h_n[-1, :, :]`).
        # Questo tensore rappresenta la comprensione della sequenza da parte del modello.
        # 3. Lo passiamo al layer lineare per ottenere i punteggi finali per ogni classe.
        out = self.fc(h_n[-1, :, :])
        return out
