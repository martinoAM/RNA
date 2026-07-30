import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoImageProcessor, AutoModelForImageClassification

from . import config
from .data_loader import build_datasets


def run_predictions(model, image_processor, dataset, device):
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    batch_size = config.BATCH_SIZE

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            pixel_values = torch.tensor(np.array(batch["pixel_values"])).to(device)
            logits = model(pixel_values=pixel_values).logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(batch["binary_label"])

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matrice de confusion - Détection de déforestation")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Matrice de confusion sauvegardée : {save_path}")


def main(model_dir=None):
    model_dir = model_dir or str(config.MODEL_OUTPUT_DIR / "final")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Chargement du modèle fine-tuné...")
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    image_processor = AutoImageProcessor.from_pretrained(model_dir)

    print("Préparation du jeu de test...")
    _, _, test_ds, id2label, label2id, _ = build_datasets()

    print("Prédiction sur le jeu de test...")
    y_true, y_pred = run_predictions(model, image_processor, test_ds, device)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    print("\n=== Résultats sur le jeu de test ===")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("\nRapport détaillé :")
    print(classification_report(y_true, y_pred, target_names=list(id2label.values())))

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=list(id2label.values()),
        save_path=str(config.FIGURES_DIR / "matrice_confusion.png"),
    )

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    main()
