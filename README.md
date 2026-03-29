# Plateforme d'analyse d'images satellitaires (multispectral & hyperspectral)

Plateforme web pour la recherche de sols rares et de gisements de fer, ainsi que l'analyse agriculture/forêts, en s'appuyant sur Google Earth Engine (GEE) et des modèles Deep Learning (ViT multibandes).

## Fonctionnalités

- Authentification via compte Google (OAuth2 — flux complet avec JWT).
- Sélection interactive d'une zone d'étude (AOI) sur carte OpenStreetMap.
- Calcul d'indices spectraux réels via GEE (NDVI, NDWI, NDBI, EVI) avec statistiques zonales et tuiles de visualisation.
- Fine-tuning d'un ViT sur images multibandes (3, 4, 10 ou 13 bandes — Sentinel-2, Landsat, drone…).
- Outil de labellisation des zones (à venir) pour constituer un dataset.
- Inference et export des résultats (tuiles, GeoJSON, raster, shapefiles — à venir).

## Architecture

```
frontend/       UI web : Leaflet (carte OSM), dessin AOI, Google Login
backend/        API FastAPI
  app/
    routers/
      auth.py       OAuth2 Google → JWT cookie httponly
      gee.py        Calcul indices via earthengine-api
      training.py   Pipeline d'entraînement (à compléter)
      inference.py  Prédiction sur nouvelles zones (à compléter)
docs/           Architecture détaillée et workflows
finetune.py     Script d'entraînement ViT multibandes (autonome)
```

## Démarrage rapide

### 1. Configuration

```bash
cd backend
cp .env.example .env
# Renseigner GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GEE_SERVICE_ACCOUNT,
# GEE_PRIVATE_KEY_PATH et JWT_SECRET dans .env
```

Générer un JWT_SECRET sécurisé :
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 2. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
# API disponible sur http://localhost:8000
# Documentation Swagger : http://localhost:8000/docs
```

### 3. Frontend

Lancer un serveur local depuis `frontend/` :
```bash
cd frontend
python -m http.server 5500
# Interface disponible sur http://localhost:5500
```

### 4. Authentification GEE

Option A — Compte de service (production) :
- Créer un compte de service sur [console.cloud.google.com](https://console.cloud.google.com)
- Activer l'API Earth Engine et télécharger la clé JSON
- Renseigner `GEE_SERVICE_ACCOUNT` et `GEE_PRIVATE_KEY_PATH` dans `.env`

Option B — Authentification utilisateur (développement local) :
```bash
earthengine authenticate
```

## API — Endpoints principaux

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/auth/google` | Redirige vers Google Login |
| `GET` | `/auth/google/callback` | Callback OAuth → JWT cookie |
| `GET` | `/auth/me` | Infos utilisateur connecté |
| `POST` | `/auth/logout` | Déconnexion |
| `POST` | `/gee/compute` | Calcul d'indice avec AOI (JSON body) |
| `GET` | `/gee/indices` | Calcul d'indice sans AOI (Maroc par défaut) |
| `POST` | `/training/run` | Lancer l'entraînement (à compléter) |
| `POST` | `/inference/run` | Lancer l'inférence (à compléter) |

Exemple de requête GEE avec AOI :
```json
POST /gee/compute
{
  "index": "NDVI",
  "collection": "SENTINEL_2",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "cloud_threshold": 20,
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[...], ...]]
  }
}
```

## Fine-tuning multibandes

```bash
pip install torch torchvision pytorch-lightning transformers rasterio albumentations
python finetune.py
```

Paramètres disponibles dans `main()` :

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `num_bands` | `4` | Nombre de bandes (3=RGB, 4=RGB+NIR, 10/13=Sentinel-2) |
| `num_classes` | `2` | Nombre de classes de sols |
| `max_epochs` | `10` | Époques d'entraînement |
| `batch_size` | `4` | Taille du batch |

Le ViT est adapté automatiquement pour accepter `num_bands` canaux tout en réutilisant les poids pré-entraînés RGB.

## Notes GEE & Hyperspectral

- Multispectral : Sentinel-2, Landsat (indices NDVI, NDWI, NDBI, EVI).
- Hyperspectral : dépend des catalogues GEE disponibles (ex. AVIRIS). Les indices minéralogiques (fer) nécessitent des combinaisons de bandes spécifiques (ex. bandes SWIR).
- Collections supportées : `SENTINEL_2`, `LANDSAT_8` (extensible dans `COLLECTION_CONFIG`).

## Sécurité

- Sessions via JWT cookie `httponly` + `samesite=lax` (protection XSS/CSRF).
- Validation de l'ID token Google côté serveur (`google.oauth2.id_token`).
- CORS à restreindre au domaine frontend en production (`allow_origins` dans `main.py`).
- Ne jamais committer le fichier `.env`.

## Prochaines étapes

- Connecter l'AOI Leaflet au endpoint `POST /gee/compute` (frontend).
- Implémenter le router `/training/run` (appel à `finetune.py`).
- Implémenter le router `/inference/run` (chargement checkpoint + prédiction).
- Ajouter l'outil de labellisation interactif (frontend).
- Ajouter une base de données (sessions, modèles, datasets).
- Mettre en place une queue de tâches (Celery/RQ) pour les jobs longs.
