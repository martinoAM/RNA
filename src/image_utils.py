"""
image_utils.py
---------------
Utilitaires de normalisation d'image, utilisés pour accepter différents
formats en entrée (PNG, JPEG, mais aussi TIFF/GeoTIFF) et les convertir
systématiquement en RGB 8 bits, quel que soit le mode d'origine.

Pourquoi c'est nécessaire pour les TIFF satellite :
- Beaucoup de TIFF issus de l'imagerie satellite sont encodés en 16 bits
  par canal (mode PIL "I;16"), alors que le modèle a été entraîné sur
  des images 8 bits classiques (0-255).
- Certains TIFF sont en niveaux de gris, en RGBA, ou multi-bandes.
Cette fonction ramène toujours l'image à un format RGB 8 bits standard
avant de la transmettre au modèle.
"""

import numpy as np
from PIL import Image, UnidentifiedImageError


def _rescale_to_uint8(array):
    """Ramène un tableau numpy quelconque (16 bits, flottant, etc.) sur
    la plage 0-255 en uint8, par simple étirement min-max."""
    array = array.astype(np.float32)
    min_val, max_val = float(array.min()), float(array.max())
    if max_val > min_val:
        array = (array - min_val) / (max_val - min_val) * 255.0
    else:
        array = np.zeros_like(array)
    return array.astype(np.uint8)


def load_image_any_format(path):
    """
    Charge une image depuis un chemin de fichier, avec repli automatique
    sur `tifffile` quand Pillow n'arrive pas à décoder le TIFF — ce qui
    arrive avec les empilements multi-bandes (ex : un TIFF Sentinel-2
    à 13, 26 bandes ou plus, issu d'un export multi-spectral ou
    multi-temporel), que Pillow limite à 4 canaux maximum (RGBA).

    Retourne toujours une image PIL en mode RGB 8 bits.
    """
    try:
        return Image.open(path)
    except (UnidentifiedImageError, OSError):
        pass

    # Repli : lecture "brute" du TIFF via tifffile, qui ne connaît pas
    # cette limite de nombre de canaux.
    import tifffile

    array = tifffile.imread(path)

    if array.ndim == 2:
        # Image mono-bande (niveaux de gris)
        return Image.fromarray(_rescale_to_uint8(array), mode="L").convert("RGB")

    if array.ndim == 3:
        # L'axe des bandes est généralement le plus petit des 3 axes
        # (ex : (26, H, W) plutôt que (H, W, 26))
        band_axis = int(np.argmin(array.shape))
        array = np.moveaxis(array, band_axis, -1)  # -> (H, W, bandes)
        n_bands = array.shape[-1]

        if n_bands >= 3:
            # Approximation : on prend les 3 premières bandes comme RGB.
            # Pour un vrai rendu couleur Sentinel-2, il faudrait plutôt
            # sélectionner précisément B4 (rouge), B3 (vert), B2 (bleu)
            # selon l'ordre réel des bandes dans le fichier.
            rgb = array[:, :, :3]
        else:
            rgb = np.repeat(array[:, :, :1], 3, axis=-1)

        return Image.fromarray(_rescale_to_uint8(rgb), mode="RGB")

    raise ValueError(f"Format TIFF non pris en charge (dimensions : {array.shape})")


def normalize_to_rgb_uint8(image: Image.Image) -> Image.Image:
    """
    Convertit une image PIL quel que soit son mode d'origine en une
    image RGB 8 bits (0-255), prête à être utilisée par le modèle.
    """
    if image.mode == "RGB":
        return image

    if image.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        # Cas fréquent avec les TIFF satellite : 16 bits ou flottant.
        # On étire le contraste sur la plage 0-255 avant de repasser en RGB.
        array = np.array(image).astype(np.float32)
        min_val, max_val = float(array.min()), float(array.max())
        if max_val > min_val:
            array = (array - min_val) / (max_val - min_val) * 255.0
        else:
            array = np.zeros_like(array)
        image = Image.fromarray(array.astype(np.uint8))

    return image.convert("RGB")
