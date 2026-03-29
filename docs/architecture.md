# Architecture détaillée

## Vue d'ensemble

```
Browser (Leaflet)
      │  HTTP/REST + cookie JWT
      ▼
FastAPI (backend/)
  ├── /auth     → Google OAuth2 + JWT
  ├── /gee      → earthengine-api (calcul indices)
  ├── /training → pipeline ML (à compléter)
  └── /inference→ prédiction (à compléter)
      │  earthengine-api
      ▼
Google Earth Engine
  ├── COPERNICUS/S2_SR_HARMONIZED  (Sentinel-2)
  └── LANDSAT/LC08/C02/T1_L2       (Landsat-8)
```

---

## Modules

### Auth (`backend/app/routers/auth.py`)

Flux OAuth2 complet avec session JWT :

1. `GET /auth/google` — génère l'URL Google et redirige l'utilisateur.
2. Google redirige vers `GET /auth/google/callback?code=...`.
3. Le backend échange le code contre un ID token Google, le valide, extrait `sub/email/name/picture`.
4. Un JWT signé (HS256, 24h) est placé dans un cookie `httponly; samesite=lax`.
5. `GET /auth/me` — décode le cookie et retourne les infos utilisateur.
6. `POST /auth/logout` — supprime le cookie.

Variables d'environnement requises : `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `OAUTH_CALLBACK_URL`, `JWT_SECRET`, `FRONTEND_URL`.

### GEE (`backend/app/routers/gee.py`)

Calcul d'indices spectraux via l'API Earth Engine :

- Initialisation via compte de service (`GEE_SERVICE_ACCOUNT` + `GEE_PRIVATE_KEY_PATH`) ou `earthengine authenticate` en développement.
- `POST /gee/compute` — endpoint principal avec AOI GeoJSON optionnelle.
- `GET /gee/indices` — alias sans AOI (zone par défaut : Maroc).

Pipeline GEE :
```
Collection → filterBounds(AOI) → filterDate → filter(nuages < seuil)
           → median() → calcul indice → reduceRegion(stats) → getMapId(tile_url)
```

Collections supportées et bandes :

| Collection | ID GEE | NIR | RED | GREEN | BLUE | SWIR |
|------------|--------|-----|-----|-------|------|------|
| SENTINEL_2 | COPERNICUS/S2_SR_HARMONIZED | B8 | B4 | B3 | B2 | B11 |
| LANDSAT_8  | LANDSAT/LC08/C02/T1_L2 | SR_B5 | SR_B4 | SR_B3 | SR_B2 | SR_B6 |

Formules des indices :

```
NDVI = (NIR - RED) / (NIR + RED)
NDWI = (GREEN - NIR) / (GREEN + NIR)
NDBI = (SWIR - NIR) / (SWIR + NIR)
EVI  = 2.5 × (NIR - RED) / (NIR + 6×RED - 7.5×BLUE + 1)
```

Réponse renvoyée par `/gee/compute` :
```json
{
  "index": "NDVI",
  "collection": "SENTINEL_2",
  "date_range": ["2023-01-01", "2023-12-31"],
  "images_count": 42,
  "statistics": { "NDVI_mean": 0.41, "NDVI_min": -0.12, "NDVI_max": 0.87, "NDVI_stdDev": 0.18 },
  "tile_url": "https://earthengine.googleapis.com/v1alpha/projects/.../maps/.../tiles/{z}/{x}/{y}"
}
```

La `tile_url` est directement utilisable dans Leaflet (`L.tileLayer`).

### Map / AOI (`frontend/`)

- Carte Leaflet centrée sur le Maroc (lat 34°, lon -6.8°, zoom 6).
- Leaflet.Draw : dessin de polygones et rectangles → GeoJSON stocké dans `currentAOI`.
- Appel `POST /gee/compute` avec l'AOI et les paramètres sélectionnés.
- Affichage de la couche de résultat via `tile_url` retournée.

### Training (`backend/app/routers/training.py` + `finetune.py`)

Pipeline d'entraînement ViT multibandes (à connecter) :

- `finetune.py` est un script autonome fonctionnel.
- `POST /training/run` doit appeler ce pipeline de manière asynchrone (task queue).

Architecture du modèle (`SatelliteClassifier`) :
```
Images (B, num_bands, 224, 224)
  └── ViT patch embedding adapté (Conv2d num_bands→hidden_dim)
       └── Transformer blocks (pré-entraînés google/vit-base-patch16-224-in21k)
            └── Token [CLS] → Linear(hidden_size, num_classes)
                 └── CrossEntropyLoss
```

Adaptation multibandes : la couche `Conv2d(3, D, 16, 16)` du patch embedding est remplacée par `Conv2d(num_bands, D, 16, 16)`. Les poids RGB sont copiés ; les bandes supplémentaires sont initialisées avec la moyenne des poids RGB.

Stats de normalisation disponibles :

| num_bands | Description |
|-----------|-------------|
| 3 | RGB (ImageNet) |
| 4 | RGB + NIR |
| 10 | Sentinel-2 (B2–B8, B8A, B11, B12) |
| 13 | Sentinel-2 toutes bandes |

### Inference (`backend/app/routers/inference.py`)

À implémenter :
- Chargement d'un checkpoint PyTorch Lightning.
- Téléchargement du patch GeoTIFF depuis GEE.
- Prédiction et retour de la classe + probabilités.

---

## Flux utilisateur

1. L'utilisateur clique "Se connecter avec Google" → redirigé vers `GET /auth/google`.
2. Après authentification, callback `GET /auth/google/callback` → JWT cookie posé.
3. L'utilisateur dessine un AOI sur la carte Leaflet.
4. Sélection : indice, collection, période → appel `POST /gee/compute`.
5. La couche résultat (tile_url) s'affiche sur la carte.
6. (À venir) Labellisation interactive → création dataset.
7. (À venir) Entraînement → `POST /training/run`.
8. (À venir) Inférence → `POST /inference/run` → export résultats.

---

## Sécurité

- Cookie JWT `httponly` + `samesite=lax` — protège contre XSS et CSRF.
- ID token Google validé côté serveur avec `google.oauth2.id_token`.
- CORS à restreindre en production (`allow_origins=[FRONTEND_URL]`).
- Secrets (clés GEE, OAuth, JWT) uniquement dans `.env`, jamais commités.

---

## Infrastructure à prévoir (production)

| Besoin | Solution suggérée |
|--------|-------------------|
| Base de données | PostgreSQL + SQLAlchemy |
| Sessions / tokens | Table `users` + JWT |
| Jobs longs (training) | Celery + Redis |
| Stockage modèles | Google Cloud Storage |
| Cache requêtes GEE | Redis |
| Monitoring | Prometheus + Grafana |
