from fastapi import APIRouter

router = APIRouter()


@router.post("/run")
def run_inference():
    # Placeholder: exécuter le modèle sur nouvelle AOI / nouvelles images
    return {"inference": "started", "status": "not_implemented"}