"""
model.py
--------
Definição da arquitetura MLP e da classe de treinamento.
Separar arquitetura de lógica de treino facilita troca de modelo
sem reescrever o loop de treinamento.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional


class MLP(nn.Module):
    """
    Rede MLP com Batch Normalization e Dropout.

    Arquitetura:
        input -> [Linear -> BN -> ReLU -> Dropout] x n_camadas -> Linear -> saída

    O uso de BN antes da ativação (estilo original Ioffe & Szegedy) estabiliza
    o treinamento ao normalizar as pré-ativações, o que geometricamente mantém
    os pontos num entorno razoável de zero antes da não-linearidade.

    Dropout age como regularização por marginalização implícita sobre um ensemble
    de sub-redes, reduzindo co-adaptação entre neurônios.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout_rate: float = 0.3
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for out_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ]
            in_dim = out_dim

        # Cabeça de classificação binária
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """
        Inicialização de Kaiming (He) para pesos das camadas lineares.
        Adequada para ReLU: mantém a variância das ativações constante ao longo
        das camadas, evitando vanishing/exploding gradients.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # shape: (batch,)


class Trainer:
    """
    Encapsula o loop de treino e validação.
    Inclui early stopping baseado na loss de validação.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # ReduceLROnPlateau: reduz LR quando a val_loss para de melhorar
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def _make_loader(self, X, y, batch_size: int, shuffle: bool) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def fit(
        self,
        X_train, y_train,
        X_val, y_val,
        epochs: int = 200,
        batch_size: int = 32,
        patience: int = 20,
        verbose: bool = True
    ):
        train_loader = self._make_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader   = self._make_loader(X_val,   y_val,   batch_size, shuffle=False)

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            # Treino
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * len(X_batch)

            train_loss /= len(X_train)

            # Validação
            val_loss, val_acc = self._evaluate(val_loader, len(X_val))
            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if verbose and epoch % 20 == 0:
                lr_atual = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:>3}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f} | "
                    f"LR: {lr_atual:.2e}"
                )

            # Early stopping
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping na época {epoch} (sem melhora por {patience} épocas).")
                    break

        # Restaura o melhor estado
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Melhor val_loss: {best_val_loss:.4f} — pesos restaurados.")

        return self.history

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, n_samples: int):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            total_loss += loss.item() * len(X_batch)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()

        return total_loss / n_samples, correct / n_samples

    @torch.no_grad()
    def predict_proba(self, X) -> torch.Tensor:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        logits = self.model(X_t)
        return torch.sigmoid(logits).cpu()
