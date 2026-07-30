from datasets import load_dataset
from transformers import AutoImageProcessor

from . import config


def load_raw_dataset():
    """
    Télécharge (ou charge depuis le cache local) le dataset EuroSAT_RGB.
    Retourne un DatasetDict Hugging Face avec un split "train" contenant
    les 27 000 images et leurs labels (10 classes).
    """
    dataset = load_dataset(config.DATASET_NAME)
    return dataset


def to_binary_forest_label(example, forest_label_id):
    """
    Convertit le label multi-classes EuroSAT en label binaire :
        1 -> "Forest"
        0 -> toute autre classe (Highway, Industrial, Residential, ...)
    """
    example["binary_label"] = int(example["label"] == forest_label_id)
    return example


def get_label_mappings(dataset):
    """
    Récupère les correspondances entre id de classe et nom de classe,
    telles que définies par le dataset EuroSAT.
    """
    features = dataset["train"].features["label"]
    id2label = {i: name for i, name in enumerate(features.names)}
    label2id = {name: i for i, name in id2label.items()}
    return id2label, label2id


def build_datasets():
    """
    Pipeline complet :
      1. Charge EuroSAT
      2. Ajoute le label binaire Forêt / Non-Forêt
      3. Découpe en train / validation / test
      4. Applique le préprocesseur d'image du modèle pré-entraîné
         (redimensionnement, normalisation identiques à l'entraînement
         d'origine du modèle -> indispensable pour le transfer learning)

    Retourne : (train_ds, val_ds, test_ds, id2label, label2id, image_processor)
    """
    raw = load_raw_dataset()
    id2label, label2id = get_label_mappings(raw)
    forest_id = label2id[config.TARGET_CLASS]

    dataset = raw["train"].map(
        lambda ex: to_binary_forest_label(ex, forest_id)
    )

    # Split train / test, puis re-split du train pour obtenir une validation
    split_1 = dataset.train_test_split(
        test_size=config.TEST_SIZE, seed=config.RANDOM_SEED
    )
    split_2 = split_1["train"].train_test_split(
        test_size=config.VAL_SIZE, seed=config.RANDOM_SEED
    )

    train_ds = split_2["train"]
    val_ds = split_2["test"]
    test_ds = split_1["test"]

    image_processor = AutoImageProcessor.from_pretrained(config.MODEL_NAME)

    def transform(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        processed = image_processor(images, return_tensors="pt")
        batch["pixel_values"] = processed["pixel_values"]
        return batch

    train_ds.set_transform(transform)
    val_ds.set_transform(transform)
    test_ds.set_transform(transform)

    binary_id2label = {0: "Non-Foret", 1: "Foret"}
    binary_label2id = {"Non-Foret": 0, "Foret": 1}

    return train_ds, val_ds, test_ds, binary_id2label, binary_label2id, image_processor


if __name__ == "__main__":
    # Petit test manuel : affiche la taille de chaque split
    train_ds, val_ds, test_ds, id2label, label2id, _ = build_datasets()
    print(f"Train : {len(train_ds)} images")
    print(f"Val   : {len(val_ds)} images")
    print(f"Test  : {len(test_ds)} images")
    print(f"Classes binaires : {id2label}")
