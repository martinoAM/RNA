import torch
from transformers import AutoModelForImageClassification

from . import config


def load_pretrained_model(id2label, label2id, freeze_backbone=True):
    """
    Charge le modèle pré-entraîné défini dans config.MODEL_NAME et
    remplace sa couche de sortie pour qu'elle corresponde à nos
    2 classes (Foret / Non-Foret) au lieu des 10 classes d'origine.

    Args:
        id2label (dict): correspondance id -> nom de classe binaire
        label2id (dict): correspondance nom de classe binaire -> id
        freeze_backbone (bool): si True, gèle les poids du "corps" du
            réseau (feature extractor) et n'entraîne que la tête de
            classification. Recommandé quand on a peu de données ou
            peu de temps de calcul.

    Returns:
        model (PreTrainedModel): modèle prêt à être fine-tuné ou évalué
    """
    model = AutoModelForImageClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            # On ne garde entraînable que la couche de classification finale
            # (son nom varie selon l'architecture : "classifier" pour
            # ConvNeXt/ViT/ResNet dans transformers)
            if "classifier" not in name:
                param.requires_grad = False

    return model


def get_embedding_model():
    """
    Charge le même modèle pré-entraîné mais configuré pour retourner les
    représentations internes (embeddings) plutôt qu'une classification.
    Utilisé par change_detection.py pour comparer deux images dans le
    temps (avant / après) sans avoir besoin de labels.
    """
    model = AutoModelForImageClassification.from_pretrained(
        config.MODEL_NAME, output_hidden_states=True
    )
    model.eval()
    return model


def count_trainable_parameters(model):
    """Utilitaire simple pour vérifier combien de poids seront réellement
    mis à jour pendant l'entraînement (utile à mentionner dans le rapport)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    id2label = {0: "Non-Foret", 1: "Foret"}
    label2id = {"Non-Foret": 0, "Foret": 1}
    model = load_pretrained_model(id2label, label2id)
    trainable, total = count_trainable_parameters(model)
    print(f"Paramètres entraînables : {trainable:,} / {total:,}")
