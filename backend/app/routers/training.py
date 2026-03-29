from fastapi import APIRouter

router = APIRouter()


@router.post("/run")
def run_training():
    # Placeholder: lancer pipeline d'entraînement (chargement dataset, model.fit)
    return {"training": "started", "status": "not_implemented"}