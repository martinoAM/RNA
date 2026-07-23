# RNA
# Détection de déforestation par vision artificielle (RNA)

Projet de réseau de neurones artificiels réutilisant un **modèle pré-entraîné existant sur Hugging Face** (transfer learning) pour détecter la déforestation à partir d'images satellite.

## 1. Contexte et sujet

Sujet choisi : **Détection de la déforestation à partir d'images satellites**.

Plutôt que d'entraîner un modèle from scratch, ce projet s'appuie sur un modèle déjà fine-tuné par la communauté Hugging Face sur le dataset **EuroSAT** (images Sentinel-2, ESA/Copernicus), que nous ré-adaptons à notre problème.

## 2. Revue des travaux existants (résumé)

- **EuroSAT** (Helber et al., 2018/2019) : dataset de référence pour la classification d'occupation du sol à partir d'images Sentinel-2, 27 000 images, 10 classes (dont "Forest").
- Plusieurs modèles pré-entraînés existent sur le Hugging Face Hub, fine-tunés sur EuroSAT avec des architectures variées (ResNet-50, ConvNeXt, Vision Transformer), atteignant ~98% d'accuracy sur la classification multi-classes.
- Les approches de détection de changement (change detection) en télédétection comparent des embeddings d'images prises à des dates différentes pour repérer les zones où un changement significatif (ex : perte de couvert forestier) a eu lieu, sans nécessiter de nouveau jeu de labels.

*(Section à enrichir avec vos propres références bibliographiques pour l'étude bibliographique complète du mémoire.)*

## 3. Jeu de données

- **Nom** : EuroSAT RGB (`blanchon/EuroSAT_RGB` sur Hugging Face)
- **Contenu** : 27 000 images satellite Sentinel-2 (64x64 px), 10 classes d'occupation du sol
- **Transformation appliquée** : le problème multi-classes est ramené à un problème binaire **Forêt / Non-Forêt**, plus directement lié à la détection de déforestation
- **Répartition** : 70% train / 10% validation / 20% test (voir `src/config.py`)

## 4. Modèle

- **Modèle de base** : `mrm8488/convnext-tiny-finetuned-eurosat` (ConvNeXt-tiny, déjà fine-tuné sur EuroSAT)
- **Méthode** : transfer learning — le corps du réseau (feature extractor) est gelé, seule la tête de classification est ré-entraînée sur la tâche binaire Forêt / Non-Forêt
- Un second module (`change_detection.py`) permet de comparer deux images de la même zone à deux dates différentes en mesurant la distance entre leurs embeddings, pour une détection de changement sans label

## 5. Structure du projet

```
rna_deforestation/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py             # constantes et hyperparamètres
│   ├── data_loader.py        # chargement et préparation d'EuroSAT
│   ├── model.py               # chargement du modèle pré-entraîné
│   ├── train.py                # fine-tuning (Trainer HF)
│   ├── evaluate.py            # métriques + matrice de confusion
│   ├── change_detection.py   # comparaison avant/après (embeddings)
│   └── demo.py                 # démonstration interactive (Gradio)
└── outputs/
    ├── modele_finetune/       # modèle sauvegardé après entraînement
    └── figures/                 # matrice de confusion, etc.
```

## 6. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 7. Utilisation

**Entraînement (fine-tuning) :**
```bash
python -m src.train
```

**Évaluation (Accuracy, Precision, Recall, F1, matrice de confusion) :**
```bash
python -m src.evaluate
```

**Détection de changement entre deux images :**
```bash
python -m src.change_detection chemin/image_avant.jpg chemin/image_apres.jpg
```

**Démonstration interactive (Gradio) :**
```bash
python -m src.demo
```
Puis ouvrir `http://127.0.0.1:7860` dans un navigateur.

## 8. Métriques d'évaluation

Le script `evaluate.py` calcule automatiquement :
- Accuracy
- Precision
- Recall
- F1-score
- Matrice de confusion (sauvegardée dans `outputs/figures/matrice_confusion.png`)

## 9. Limites et pistes d'amélioration

- EuroSAT est un dataset européen : les résultats devront être validés sur des images satellite de Madagascar (biais de distribution possible entre forêts tempérées européennes et forêts tropicales malgaches).
- La résolution EuroSAT (10m/pixel, tuiles 64x64) limite la précision de détection à l'échelle de petites parcelles ; une imagerie à plus haute résolution (Planet, SPOT) améliorerait la finesse de détection.
- Le seuil de détection de changement (`CHANGE_DETECTION_THRESHOLD`) est fixé empiriquement et devrait être calibré sur un jeu de paires avant/après annotées.
