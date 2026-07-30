import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments

from . import config
from .data_loader import build_datasets
from .model import load_pretrained_model


def collate_fn(batch):
    """Assemble une liste d'exemples en un batch de tenseurs PyTorch."""
    pixel_values = torch.stack([torch.tensor(x["pixel_values"]) for x in batch])
    labels = torch.tensor([x["binary_label"] for x in batch])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    """
    Calcule les métriques demandées dans le cahier des charges :
    Accuracy, Precision, Recall, F1-score.
    La matrice de confusion complète est produite séparément dans
    evaluate.py (plus lisible sous forme de figure).
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    print("Chargement du dataset EuroSAT et préparation des splits...")
    train_ds, val_ds, test_ds, id2label, label2id, image_processor = build_datasets()

    print("Chargement du modèle pré-entraîné (transfer learning)...")
    model = load_pretrained_model(id2label, label2id, freeze_backbone=True)

    training_args = TrainingArguments(
        output_dir=str(config.MODEL_OUTPUT_DIR),
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        remove_unused_columns=False,
        seed=config.RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        image_processor=image_processor,
    )

    print("Début du fine-tuning...")
    trainer.train()

    print("Sauvegarde du modèle final...")
    trainer.save_model(str(config.MODEL_OUTPUT_DIR / "final"))
    image_processor.save_pretrained(str(config.MODEL_OUTPUT_DIR / "final"))

    print("Évaluation finale sur le jeu de validation :")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return trainer, test_ds


if __name__ == "__main__":
    main()
