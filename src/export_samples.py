import argparse
from collections import defaultdict

from . import config
from .data_loader import load_raw_dataset, get_label_mappings


def export_samples(n_par_classe=3):
    print("Chargement du dataset EuroSAT (peut prendre un moment la 1ère fois)...")
    raw = load_raw_dataset()
    id2label, _ = get_label_mappings(raw)

    output_dir = config.OUTPUTS_DIR / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    compteur = defaultdict(int)
    total_a_exporter = n_par_classe * len(id2label)
    exportees = 0

    for example in raw["train"]:
        label_name = id2label[example["label"]]

        if compteur[label_name] >= n_par_classe:
            continue

        image = example["image"].convert("RGB")
        nom_fichier = f"{label_name}_{compteur[label_name] + 1}.jpg"
        chemin = output_dir / nom_fichier
        image.save(chemin, "JPEG")

        compteur[label_name] += 1
        exportees += 1

        if exportees >= total_a_exporter:
            break

    print(f"\n{exportees} images exportées dans : {output_dir}")
    for classe, n in compteur.items():
        print(f"  - {classe} : {n} image(s)")
    print(
        "\nTu peux maintenant les uploader une par une dans la démo "
        "(python3 -m src.demo), ou utiliser deux images 'Forest' vs "
        "'Highway'/'Industrial' pour tester la détection de changement."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporte des images EuroSAT pour test manuel")
    parser.add_argument(
        "--par-classe", type=int, default=3,
        help="Nombre d'images à exporter par classe (défaut : 3)"
    )
    args = parser.parse_args()
    export_samples(n_par_classe=args.par_classe)
