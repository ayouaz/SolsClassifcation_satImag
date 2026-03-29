from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import auth, gee, training, inference

app = FastAPI(title="SatImag Platform API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(gee.router, prefix="/gee", tags=["gee"])
app.include_router(training.router, prefix="/training", tags=["training"])
app.include_router(inference.router, prefix="/inference", tags=["inference"])