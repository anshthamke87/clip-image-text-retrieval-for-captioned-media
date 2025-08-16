# CLIP Image-Text Retrieval Engine for Captioned Media

A complete end-to-end implementation of bi-directional image-text retrieval using OpenCLIP, FAISS indexing, and production FastAPI deployment.

## 🎯 Project Overview

This project implements a complete production pipeline for building a scalable image-text retrieval system that can:
- **Search images using text queries** (text → image retrieval)
- **Find relevant captions for images** (image → text retrieval)
- **Scale to large datasets** using efficient ANN indexing
- **Deploy in production** with FastAPI and Docker

## 🏆 Final Results

### Production System Performance
**Complete end-to-end system deployed with:**

| Component | Performance | Technology |
|-----------|-------------|------------|
| **Model** | 50.7% text→image R@1, 70.0% image→text R@1 | OpenCLIP ViT-B-32 |
| **Search** | <1ms per query | FAISS HNSW |
| **API** | 8 endpoints, auto-docs | FastAPI |
| **Deployment** | Production-ready | Docker + docker-compose |

### Benchmark Comparison
| Direction | Recall@1 | Recall@5 | Recall@10 | Query Speed |
|-----------|----------|----------|-----------|-------------|
| Text → Image | **50.7%** | 76.9% | 85.8% | 0.1ms |
| Image → Text | **70.0%** | 89.1% | 94.9% | 0.2ms |

*These results are competitive with published academic papers while maintaining production-grade performance.*

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   OpenCLIP       │    │   FAISS HNSW    │
│   Web Service   │───▶│   ViT-B/32       │───▶│   Index Search  │
│   (8 endpoints) │    │   (50.7% R@1)    │    │   (<1ms query)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Docker        │    │   Sub-millisecond│    │   8,091 Images  │
│   Deployment    │    │   Embeddings     │    │   40,455 Captions│
│   (Production)  │    │   (512-dim)      │    │   (Flickr8k)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📊 Dataset

**Flickr8k (Professionally Processed)**
- **Images**: 8,091 unique images (validated and organized)
- **Captions**: 40,455 total captions (5 per image, normalized)
- **Splits**: 70% train (5,663) / 15% val (1,213) / 15% test (1,215)
- **Quality**: 100% image validation, comprehensive error handling

## 🎛️ Complete Project Pipeline

### ✅ Phase 1: Data Engineering (COMPLETE)
- [x] **Dataset Processing**: Flickr8k download, validation, and cleaning
- [x] **Split Creation**: Proper train/validation/test splits by unique images
- [x] **Quality Assurance**: Image integrity validation and caption normalization
- [x] **Documentation**: Comprehensive dataset documentation with checksums

### ✅ Phase 2: Baseline Establishment (COMPLETE)
- [x] **Model Integration**: OpenCLIP ViT-B-32 implementation
- [x] **Evaluation Pipeline**: Comprehensive Recall@K and ranking metrics
- [x] **Performance Achievement**: 50.7% text→image and 70.0% image→text R@1
- [x] **Embedding Storage**: Efficient caching for downstream tasks

### ✅ Phase 3: Scalability Optimization (COMPLETE)
- [x] **FAISS Integration**: HNSW index for approximate nearest neighbor search
- [x] **Parameter Tuning**: Systematic optimization of accuracy vs speed trade-offs
- [x] **Performance Validation**: Sub-millisecond queries with zero quality loss
- [x] **Production Readiness**: Memory-efficient indexes ready for deployment

### ✅ Phase 4: Advanced Training Exploration (COMPLETE)
- [x] **Fine-tuning Attempts**: Multiple approaches including conservative training
- [x] **Challenge Documentation**: Comprehensive analysis of catastrophic forgetting
- [x] **Production Decision**: Evidence-based choice of baseline model
- [x] **Lessons Learned**: Valuable insights into vision-language model training

### ✅ Phase 5: Production Deployment (COMPLETE)
- [x] **FastAPI Service**: Professional REST API with 8 endpoints
- [x] **Request Validation**: Pydantic models with comprehensive error handling
- [x] **Performance Monitoring**: Built-in statistics and health checks
- [x] **Container Deployment**: Docker and docker-compose for production

## 🚀 Quick Start

### 🐳 Production Deployment
```bash
# Clone repository
git clone https://github.com/anshthamke87/clip-image-text-retrieval-for-captioned-media.git
cd clip-image-text-retrieval-for-captioned-media

# Deploy with Docker
cd service
docker-compose up --build

# Access API
curl http://localhost:8000/health
```

### 📚 API Documentation
Once deployed, access:
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### 🔍 Usage Examples
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

# Text encoding
response = requests.post(
    "http://localhost:8000/encode/text",
    json={"text": "example caption"}
)
```

## 📁 Project Structure

```
├── data/                          # Dataset and processing
│   ├── train.jsonl               # Training data (28,315 entries)
│   ├── val.jsonl                 # Validation data (6,065 entries)
│   ├── test.jsonl                # Test data (6,075 entries)
│   ├── images/                   # Processed images (8,091 files)
│   └── README.md                 # Dataset documentation
├── artifacts/                    # Models and computed assets
│   ├── embeddings/              # Pre-computed embeddings
│   ├── indexes/                 # FAISS search indexes
│   │   ├── image_hnsw_index.faiss
│   │   ├── text_hnsw_index.faiss
│   │   └── index_metadata.json
│   ├── models/                  # Model checkpoints
│   └── project_state.json       # Project tracking
├── results/                     # Evaluation and analysis
│   ├── zero_shot_baseline_results.json
│   ├── faiss_parameter_tuning.json
│   ├── final_model_results.json
│   └── production_api_results.json
├── reports/                     # Visualizations and analysis
│   ├── faiss_tradeoff_analysis.png
│   └── baseline_vs_finetuning_analysis.png
├── service/                     # Production deployment
│   ├── main.py                  # FastAPI application
│   ├── requirements.txt         # Dependencies
│   ├── Dockerfile              # Container definition
│   ├── docker-compose.yml      # Orchestration
│   ├── config.json             # Service configuration
│   └── README.md               # Deployment guide
└── README.md                    # This file
```

## 🔧 Technical Implementation

### Model Architecture
- **Vision Encoder**: OpenCLIP ViT-B/32 (86M parameters)
- **Text Encoder**: Transformer with 63M parameters  
- **Embedding Dimension**: 512-dimensional L2-normalized vectors
- **Training**: Pre-trained on 400M image-text pairs

### Search Infrastructure
- **Index Type**: Hierarchical Navigable Small World (HNSW)
- **Parameters**: M=16, efConstruction=200, optimized efSearch
- **Memory Usage**: <50MB for complete search indexes
- **Throughput**: >1000 queries per second per core

### API Features
- **Endpoints**: 8 RESTful endpoints with full OpenAPI documentation
- **Validation**: Comprehensive request/response validation with Pydantic
- **Caching**: Intelligent query result caching for performance
- **Monitoring**: Built-in health checks and usage statistics
- **Error Handling**: Professional error responses with detailed logging

## 📈 Performance Analysis

### Benchmark Results
Our system achieves competitive performance with academic state-of-the-art while maintaining production requirements:

- **Accuracy**: 50.7% text→image R@1 (competitive with published papers)
- **Speed**: Sub-millisecond search (production-grade performance)
- **Scalability**: Linear scaling with dataset size
- **Reliability**: Comprehensive error handling and monitoring

### Production Characteristics
- **Startup Time**: <30 seconds including model loading
- **Memory Usage**: <2GB RAM for complete system
- **CPU Efficiency**: Optimized for CPU-only deployment
- **Concurrency**: Async FastAPI handles multiple concurrent requests

## 🎓 Key Learning Outcomes

This project demonstrates mastery of:

### **Technical Skills**
- **Multimodal AI**: Vision-language model implementation and optimization
- **Information Retrieval**: Building efficient similarity search systems
- **ML Engineering**: End-to-end pipeline from data to deployment
- **API Development**: Production-grade REST service design
- **Containerization**: Docker-based deployment strategies

### **Engineering Practices**
- **Data Engineering**: Large-scale dataset processing and validation
- **Performance Optimization**: Systematic accuracy vs speed trade-offs
- **Production Deployment**: Complete CI/CD pipeline considerations
- **Documentation**: Comprehensive technical and user documentation
- **Testing**: Systematic validation and error handling

### **Research Insights**
- **Baseline Strength**: Pre-trained models often outperform fine-tuning
- **Fine-tuning Challenges**: Catastrophic forgetting in vision-language models
- **Production Trade-offs**: Balancing accuracy, speed, and reliability
- **System Design**: Building scalable ML systems from research prototypes

## 🔬 Research Contributions

### Novel Findings
- **Baseline Efficacy**: Demonstrated that OpenCLIP baseline (50.7% R@1) is competitive and stable
- **Fine-tuning Sensitivity**: Documented challenges with CLIP fine-tuning on domain-specific data
- **Production Optimization**: Systematic analysis of FAISS parameter optimization for CLIP embeddings

### Methodological Contributions
- **Evaluation Pipeline**: Comprehensive bi-directional retrieval evaluation
- **Deployment Framework**: Complete pipeline from research to production
- **Performance Analysis**: Detailed latency vs accuracy characterization

## 📊 Future Enhancements

### Technical Improvements
- **Model Scaling**: Experiment with larger CLIP variants (ViT-L, ViT-H)
- **Multilingual Support**: Extend to non-English captions and queries
- **Domain Adaptation**: Explore safe fine-tuning approaches for specific domains
- **Real-time Learning**: Implement online learning for user feedback

### System Enhancements
- **Auto-scaling**: Kubernetes deployment with horizontal scaling
- **Advanced Caching**: Redis-based distributed caching
- **Monitoring**: Comprehensive logging and metrics with Prometheus
- **Security**: Authentication, rate limiting, and input sanitization

## 📧 Contact

**Ansh Thamke**
- Email: at3841@columbia.edu
- GitHub: [@anshthamke87](https://github.com/anshthamke87)
- LinkedIn: [Connect for collaboration opportunities]

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@software{thamke2024clip_retrieval,
  title={CLIP Image-Text Retrieval Engine for Captioned Media},
  author={Thamke, Ansh},
  year={2024},
  url={https://github.com/anshthamke87/clip-image-text-retrieval-for-captioned-media},
  note={Complete end-to-end implementation with production deployment}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**🎉 Built with ❤️ for advancing multimodal AI systems and production ML deployment**

*This project represents a complete journey from research idea to production deployment, demonstrating both technical depth and engineering excellence.*
