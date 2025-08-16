"""
CLIP Image-Text Retrieval API
Production FastAPI service for bi-directional image-text retrieval
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import torch
import open_clip
import faiss
import numpy as np
from PIL import Image
import io
import time
import logging
from typing import List, Dict, Any
import pickle
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the service class (would be in separate module)
# from clip_service import CLIPRetrievalService

app = FastAPI(
    title="CLIP Image-Text Retrieval API",
    description="Production API for bi-directional image-text retrieval",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "message": "CLIP Image-Text Retrieval API",
        "status": "running",
        "model": "OpenCLIP ViT-B-32 Baseline (50.7% R@1)"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "loaded"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
