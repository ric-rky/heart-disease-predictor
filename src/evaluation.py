"""
evaluation.py
-------------
Funções de avaliação do modelo: métricas, curvas e visualizações.
Separado do model.py para poder ser reaproveitado com qualquer classificador.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay
)


def print_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5):
    """
    Imprime relatório de classificação, AUC-ROC e Average Precision.
    """
    y_pred = (y_proba >= threshold).astype(int)

    auc   = roc_auc_score(y_true, y_proba)
    ap    = average_precision_score(y_true, y_proba)

    print("=" * 50)
    print(f"AUC-ROC:           {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=["Sem doença", "Com doença"]))


def plot_training_history(history: dict, figsize=(12, 4)):
    """
    Plota as curvas de loss e acurácia ao longo das épocas.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Treino", color="#2563EB")
    axes[0].plot(epochs, history["val_loss"],   label="Validação", color="#DC2626", linestyle="--")
    axes[0].set_title("Loss por Época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_acc"], label="Acurácia (Val)", color="#16A34A")
    axes[1].set_title("Acurácia de Validação por Época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Histórico de Treinamento", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_evaluation(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5):
    """
    Painel com matriz de confusão, curva ROC e curva Precision-Recall.
    """
    y_pred = (y_proba >= threshold).astype(int)

    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    # Matriz de confusão
    ax1 = fig.add_subplot(gs[0])
    cm  = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Sem doença", "Com doença"])
    disp.plot(ax=ax1, colorbar=False, cmap="Blues")
    ax1.set_title("Matriz de Confusão", fontweight="bold")

    # Curva ROC
    ax2 = fig.add_subplot(gs[1])
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    ax2.plot(fpr, tpr, color="#2563EB", lw=2, label=f"AUC = {auc:.3f}")
    ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax2.set_xlabel("FPR (1 - Especificidade)")
    ax2.set_ylabel("TPR (Sensibilidade)")
    ax2.set_title("Curva ROC", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Curva Precision-Recall
    ax3 = fig.add_subplot(gs[2])
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    ax3.plot(rec, prec, color="#DC2626", lw=2, label=f"AP = {ap:.3f}")
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.set_title("Curva Precision-Recall", fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle("Avaliação no Conjunto de Teste", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_threshold_analysis(y_true: np.ndarray, y_proba: np.ndarray):
    """
    Plota como F1, Precision e Recall variam com o threshold de decisão.
    Útil para ajustar o threshold conforme o custo de FP vs FN no contexto clínico.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    thresholds = np.linspace(0.1, 0.9, 80)
    f1s, precs, recs = [], [], []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))

    best_t = thresholds[np.argmax(f1s)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, f1s,   label="F1",        color="#2563EB",  lw=2)
    ax.plot(thresholds, precs, label="Precision",  color="#16A34A",  lw=2)
    ax.plot(thresholds, recs,  label="Recall",     color="#DC2626",  lw=2)
    ax.axvline(best_t, color="black", linestyle="--", alpha=0.5,
               label=f"Melhor threshold (F1): {best_t:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Análise de Threshold", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Threshold que maximiza F1: {best_t:.3f} (F1 = {max(f1s):.4f})")
    return best_t
