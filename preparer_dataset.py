
from pathlib import Path
import random
import shutil

# Dossier contenant les images téléchargées
SOURCE = Path("Vehicle-Re-ID-Dataset-main")

# Dossier de destination
DESTINATION = Path("dataset")

# Extensions d'images acceptées
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Répartition reproductible
random.seed(42)

# Mots-clés permettant de reconnaître les catégories
CATEGORIES = {
    "voitures": ["car", "sedan", "suv", "van"],
    "motos": ["motorcycle", "motorbike", "bike"],
}

if not SOURCE.exists():
    raise FileNotFoundError(f"Dossier introuvable : {SOURCE.resolve()}")

images = [
    p for p in SOURCE.rglob("*")
    if p.is_file() and p.suffix.lower() in EXTENSIONS
]

selection = {"voitures": [], "motos": []}

for image in images:
    nom = image.name.lower()

    for categorie, mots_cles in CATEGORIES.items():
        if any(mot in nom for mot in mots_cles):
            selection[categorie].append(image)
            break

for categorie, fichiers in selection.items():
    random.shuffle(fichiers)

    limite = int(len(fichiers) * 0.8)
    repartition = {
        "train": fichiers[:limite],
        "validation": fichiers[limite:],
    }

    for ensemble, liste in repartition.items():
        dossier = DESTINATION / ensemble / categorie
        dossier.mkdir(parents=True, exist_ok=True)

        for i, fichier in enumerate(liste):
            extension = fichier.suffix.lower()
            cible = dossier / f"{categorie}_{i}{extension}"
            shutil.copy2(fichier, cible)

    print(
        f"{categorie} : {len(fichiers)} images "
        f"({len(repartition['train'])} train, "
        f"{len(repartition['validation'])} validation)"
    )

print("\nPréparation terminée.")
print(f"Dossier créé : {DESTINATION.resolve()}")
