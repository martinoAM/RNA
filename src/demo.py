from . import _gradio_patch

import gradio as gr
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from . import config
from .change_detection import compare_images
from .image_utils import normalize_to_rgb_uint8, load_image_any_format

MODEL_PATH = config.MODEL_OUTPUT_DIR / "final"
MODEL_PATH = str(MODEL_PATH) if MODEL_PATH.exists() else config.MODEL_NAME

model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model.eval()


def classify_image(image_path):
    if image_path is None:
        return {}
    image = load_image_any_format(image_path)
    image = normalize_to_rgb_uint8(image)
    inputs = image_processor(image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).squeeze().tolist()

    if isinstance(probs, float):
        probs = [probs]

    labels = model.config.id2label
    return {labels[i]: float(probs[i]) for i in range(len(probs))}


def detect_change(image_before, image_after):
    result = compare_images(image_before, image_after)
    return (
        f"Distance calculée : {result['distance']}\n"
        f"Seuil : {result['seuil']}\n"
        f"Résultat : {result['interpretation']}"
    )


with gr.Blocks(title="Détection de déforestation - Démo RNA") as demo:
    gr.Markdown(
        "# Détection de déforestation à partir d'images satellite\n"
        "Projet RNA basé sur un modèle pré-entraîné du Hugging Face Hub, "
        "fine-tuné sur le dataset EuroSAT (Sentinel-2)."
    )

    with gr.Tab("Classification d'une image"):
        gr.Markdown(
            "Envoie une image satellite pour savoir si la zone est de la forêt ou non. "
            "Formats acceptés : PNG, JPEG, TIFF/GeoTIFF (y compris 16 bits)."
        )
        img_input = gr.File(
            label="Image satellite (PNG, JPEG, TIFF/GeoTIFF)",
            file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
            type="filepath",
        )
        classify_btn = gr.Button("Classifier")
        classify_output = gr.Label(label="Résultat")
        classify_btn.click(fn=classify_image, inputs=img_input, outputs=classify_output)

    with gr.Tab("Détection de changement (avant / après)"):
        gr.Markdown(
            "Envoie deux images satellite de la même zone, prises à des "
            "dates différentes, pour détecter un éventuel changement "
            "(ex : déforestation). Formats acceptés : PNG, JPEG, TIFF/GeoTIFF."
        )
        with gr.Row():
            img_before = gr.File(
                label="Image avant", file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff"], type="filepath"
            )
            img_after = gr.File(
                label="Image après", file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff"], type="filepath"
            )
        change_btn = gr.Button("Comparer")
        change_output = gr.Textbox(label="Résultat de l'analyse", lines=4)
        change_btn.click(
            fn=detect_change, inputs=[img_before, img_after], outputs=change_output
        )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_api=False)