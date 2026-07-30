from pathlib import Path

# ---------------------------------------------------------------------------
# Modèle pré-entraîné utilisé comme point de départ (transfer learning)
# ---------------------------------------------------------------------------
# Ces modèles ont déjà été fine-tunés sur EuroSAT et sont disponibles
# publiquement sur le Hugging Face Hub :
#   - "mrm8488/convnext-tiny-finetuned-eurosat"  -> ConvNeXt-tiny, ~98.5% acc
#   - "cm93/resnet50-eurosat"                     -> ResNet-50
MODEL_NAME = "mrm8488/convnext-tiny-finetuned-eurosat"

# ---------------------------------------------------------------------------
# Dataset : EuroSAT (images Sentinel-2, 10 classes d'occupation du sol)
# ---------------------------------------------------------------------------
DATASET_NAME = "blanchon/EuroSAT_RGB"

# La classe qui nous intéresse pour la détection de déforestation.
# EuroSAT ne fournit pas directement un label "déforestation" : on
# transforme donc le problème en "Forest" vs "Non-Forest", ce qui permet
# ensuite de détecter un changement d'état entre deux dates (avant/après).
TARGET_CLASS = "Forest"

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_OUTPUT_DIR = OUTPUTS_DIR / "modele_finetune"
FIGURES_DIR = OUTPUTS_DIR / "figures"

for d in (OUTPUTS_DIR, MODEL_OUTPUT_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparamètres d'entraînement (fine-tuning)
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
TEST_SIZE = 0.2          # proportion réservée au test
VAL_SIZE = 0.1           # proportion réservée à la validation
RANDOM_SEED = 42

# Seuil de distance utilisé par le module de détection de changement
# (change_detection.py). Deux embeddings dont la distance cosinus dépasse
# ce seuil sont considérés comme représentant un changement significatif
# (ex : forêt -> sol nu / zone déforestée).
CHANGE_DETECTION_THRESHOLD = 0.15
