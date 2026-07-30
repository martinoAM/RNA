import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor

from . import config
from .model import get_embedding_model
from .image_utils import normalize_to_rgb_uint8, load_image_any_format


def extract_embedding(image_path, model, image_processor):
    """Retourne le vecteur d'embedding (pooled) d'une image satellite."""
    image = load_image_any_format(image_path)
    image = normalize_to_rgb_uint8(image)
    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden = outputs.hidden_states[-1]
    if last_hidden.dim() == 4:
        embedding = last_hidden.mean(dim=[2, 3])
    else:
        embedding = last_hidden.mean(dim=1)

    return embedding.squeeze(0)


def compare_images(image_before_path, image_after_path, threshold=None):
    """
    Compare deux images satellite de la même zone (avant/après) et
    détermine si un changement significatif (potentielle déforestation)
    a eu lieu.

    Returns:
        dict avec la distance calculée, le seuil utilisé, et la décision.
    """
    threshold = threshold or config.CHANGE_DETECTION_THRESHOLD

    model = get_embedding_model()
    image_processor = AutoImageProcessor.from_pretrained(config.MODEL_NAME)

    emb_before = extract_embedding(image_before_path, model, image_processor)
    emb_after = extract_embedding(image_after_path, model, image_processor)

    cosine_sim = F.cosine_similarity(emb_before.unsqueeze(0), emb_after.unsqueeze(0)).item()
    distance = 1 - cosine_sim

    changement_detecte = distance > threshold

    return {
        "distance": round(distance, 4),
        "seuil": threshold,
        "changement_detecte": changement_detecte,
        "interpretation": (
            "Changement significatif détecté (possible déforestation)"
            if changement_detecte
            else "Pas de changement significatif détecté"
        ),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage : python -m src.change_detection <image_avant> <image_apres>")
        sys.exit(1)

    result = compare_images(sys.argv[1], sys.argv[2])
    print(result)
