# CLIP Retrieval API Deployment

## Quick Start

### Local Development
```bash
cd service
pip install -r requirements.txt
python main.py
```

### Docker Deployment
```bash
cd service
docker-compose up --build
```

### API Usage
```python
import requests

# Text-to-image search
response = requests.post(
    "http://localhost:8000/search/images",
    json={"text": "a dog playing in water", "k": 5}
)

# Image-to-text search
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/search/text",
        files={"file": f},
        params={"k": 5}
    )
```

## Performance
- **Model**: OpenCLIP ViT-B-32 baseline (50.7% text→image R@1)
- **Search Speed**: <1ms per query with FAISS HNSW
- **Scalability**: Production-ready with caching and validation

## Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `POST /search/images` - Text→image retrieval
- `POST /search/text` - Image→text retrieval
- `POST /encode/text` - Text encoding
- `POST /encode/image` - Image encoding
