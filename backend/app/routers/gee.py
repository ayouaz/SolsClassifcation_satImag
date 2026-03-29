from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import ee
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

_gee_initialized = False


def _init_gee():
    global _gee_initialized
    if _gee_initialized:
        return
    service_account = os.getenv("GEE_SERVICE_ACCOUNT")
    key_path = os.getenv("GEE_PRIVATE_KEY_PATH")
    if service_account and key_path and os.path.exists(key_path):
        credentials = ee.ServiceAccountCredentials(service_account, key_path)
        ee.Initialize(credentials)
    else:
        try:
            ee.Initialize()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=(
                    "GEE non initialisé. Configurez GEE_SERVICE_ACCOUNT et "
                    f"GEE_PRIVATE_KEY_PATH dans .env, ou lancez 'earthengine authenticate'. Erreur: {e}"
                ),
            )
    _gee_initialized = True


# Configuration des collections satellite
COLLECTION_CONFIG = {
    "SENTINEL_2": {
        "id": "COPERNICUS/S2_SR_HARMONIZED",
        "bands": {"blue": "B2", "green": "B3", "red": "B4", "nir": "B8", "swir": "B11"},
        "scale": 10,
        "cloud_property": "CLOUDY_PIXEL_PERCENTAGE",
    },
    "LANDSAT_8": {
        "id": "LANDSAT/LC08/C02/T1_L2",
        "bands": {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4", "nir": "SR_B5", "swir": "SR_B6"},
        "scale": 30,
        "cloud_property": "CLOUD_COVER",
    },
}

VALID_INDICES = ["NDVI", "NDWI", "NDBI", "EVI"]


class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: list


class IndexRequest(BaseModel):
    index: str
    collection: str = "SENTINEL_2"
    start_date: str = "2022-01-01"
    end_date: str = "2022-12-31"
    cloud_threshold: int = 20
    geometry: Optional[GeoJSONGeometry] = None


def _compute_index_image(image: ee.Image, index: str, bands: dict) -> ee.Image:
    """Calcule l'image d'un indice spectral à partir des bandes sélectionnées."""
    b = {k: image.select(v) for k, v in bands.items()}
    if index == "NDVI":
        return b["nir"].subtract(b["red"]).divide(b["nir"].add(b["red"])).rename("NDVI")
    if index == "NDWI":
        return b["green"].subtract(b["nir"]).divide(b["green"].add(b["nir"])).rename("NDWI")
    if index == "NDBI":
        return b["swir"].subtract(b["nir"]).divide(b["swir"].add(b["nir"])).rename("NDBI")
    if index == "EVI":
        return (
            b["nir"].subtract(b["red"]).multiply(2.5)
            .divide(
                b["nir"]
                .add(b["red"].multiply(6))
                .subtract(b["blue"].multiply(7.5))
                .add(1)
            )
            .rename("EVI")
        )
    raise ValueError(f"Indice inconnu: {index}")


@router.post("/compute")
def compute_index(req: IndexRequest):
    """Calcule un indice spectral (NDVI, NDWI, NDBI, EVI) via GEE sur une AOI."""
    _init_gee()

    index = req.index.upper()
    if index not in VALID_INDICES:
        raise HTTPException(status_code=400, detail=f"Indice invalide. Choisissez parmi: {VALID_INDICES}")

    config = COLLECTION_CONFIG.get(req.collection)
    if not config:
        raise HTTPException(
            status_code=400,
            detail=f"Collection inconnue. Choisissez parmi: {list(COLLECTION_CONFIG.keys())}",
        )

    # Zone d'intérêt : AOI fournie ou Maroc par défaut
    if req.geometry:
        aoi = ee.Geometry({"type": req.geometry.type, "coordinates": req.geometry.coordinates})
    else:
        aoi = ee.Geometry.BBox(-17.0, 21.0, -1.0, 36.0)

    # Filtrage de la collection
    collection = (
        ee.ImageCollection(config["id"])
        .filterBounds(aoi)
        .filterDate(req.start_date, req.end_date)
        .filter(ee.Filter.lt(config["cloud_property"], req.cloud_threshold))
    )

    count = collection.size().getInfo()
    if count == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Aucune image trouvée pour {req.start_date} → {req.end_date} "
                f"avec moins de {req.cloud_threshold}% de nuages."
            ),
        )

    # Médiane temporelle pour réduire les nuages résiduels
    median_image = collection.median()
    index_image = _compute_index_image(median_image, index, config["bands"])

    # Statistiques sur l'AOI
    stats = index_image.reduceRegion(
        reducer=ee.Reducer.mean()
            .combine(ee.Reducer.min(), sharedInputs=True)
            .combine(ee.Reducer.max(), sharedInputs=True)
            .combine(ee.Reducer.stdDev(), sharedInputs=True),
        geometry=aoi,
        scale=config["scale"],
        maxPixels=1e9,
    ).getInfo()

    # URL de tuile pour visualisation Leaflet
    palette = {
        "NDVI": {"min": -0.2, "max": 0.8,  "palette": ["brown", "white", "darkgreen"]},
        "NDWI": {"min": -0.5, "max": 0.5,  "palette": ["brown", "white", "blue"]},
        "NDBI": {"min": -0.5, "max": 0.5,  "palette": ["green", "white", "red"]},
        "EVI":  {"min": -0.2, "max": 0.8,  "palette": ["brown", "white", "darkgreen"]},
    }
    map_id = index_image.getMapId(palette[index])
    tile_url = map_id["tile_fetcher"].url_format

    return {
        "index": index,
        "collection": req.collection,
        "date_range": [req.start_date, req.end_date],
        "images_count": count,
        "statistics": stats,
        "tile_url": tile_url,
    }


# Compatibilité avec l'ancien endpoint GET (sans AOI)
@router.get("/indices")
def compute_index_get(
    index: str = "NDVI",
    collection: str = "SENTINEL_2",
    start_date: str = "2022-01-01",
    end_date: str = "2022-12-31",
):
    """Alias GET pour /compute (AOI par défaut = Maroc)."""
    return compute_index(
        IndexRequest(index=index, collection=collection, start_date=start_date, end_date=end_date)
    )
