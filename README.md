---
title: Brain Tumor Detection API
emoji: 🧠
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Brain Tumor Detection API

API FastAPI qui sert un ResNet18 fine-tuné pour classifier les IRM cérébrales en 4 classes (`glioma`, `meningioma`, `pituitary`, `notumor`), avec carte d'explicabilité Grad-CAM et génération automatique de rapport.

## Avant le premier déploiement

Ce Space a besoin du fichier de poids du modèle, **`best_model.pth`** (~45 Mo), qui n'est pas inclus ici (trop volumineux pour être généré automatiquement). Ajoute-le à la racine de ce Space via l'onglet **Files** → **Add file** → **Upload files**, en le prenant sur ton ordinateur (celui téléchargé depuis Colab).

## Endpoints

- `GET /` — vérifie que l'API est en ligne
- `POST /predict` — reçoit un fichier image (`file`), renvoie la classification, les probabilités par classe, la carte Grad-CAM encodée en base64, et un rapport texte structuré

## Test rapide

```bash
curl -X POST "https://<ton-espace>.hf.space/predict" -F "file=@image.jpg"
```
