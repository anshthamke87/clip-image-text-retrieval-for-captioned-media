# CLIP Image-Text Retrieval Engine for Captioned Media

A comprehensive implementation of bi-directional image-text retrieval using OpenCLIP, FAISS indexing, and fine-tuning techniques.

## 🎯 Project Overview

This project implements a complete pipeline for building a scalable image-text retrieval system that can:
- **Search images using text queries** (text → image retrieval)
- **Find relevant captions for images** (image → text retrieval)
- **Scale to large datasets** using efficient ANN indexing
- **Improve through fine-tuning** with ranking losses

## 📊 Current Results

### Zero-Shot Baseline (OpenCLIP ViT-B/32)
**Flickr8k Test Set Performance:**

| Direction | Recall@1 | Recall@5 | Recall@10 |
|-----------|----------|----------|-----------|
| Text → Image | 50.7% | 76.9% | 85.8% |
| Image → Text | 70.0% | 89.1% | 94.9% |

### Efficient Search (FAISS HNSW)
**Production-Ready Performance:**

| Direction | Recall@1 | Query Speed | Speedup |
|-----------|----------|-------------|---------|
| Text → Image | 50.7% | 0.1ms | 6-9x faster |
| Image → Text | 70.0% | 0.2ms | 4-6x faster |

*Perfect quality preservation with sub-millisecond response times!*

## 🏗️ Architecture

- **Base Model**: OpenCLIP ViT-B/32 (pre-trained vision-language model)
- **Indexing**: FAISS HNSW for approximate nearest neighbor search
- **Search Speed**: Sub-millisecond queries for real-time applications
- **Quality**: Zero degradation vs brute-force search
- **Dataset**: Flickr8k (8,091 images, 40,455 captions)
- **API**: FastAPI service for real-time retrieval (next phase)

## 📊 Dataset

**Flickr8k (Processed)**
- **Images**: 8,091 unique images
- **Captions**: 40,455 total captions (5 per image)
- **Splits**: 70% train (5,663) / 15% val (1,213) / 15% test (1,215)
- **Format**: Clean JSONL files with proper image-caption alignment

## 🎛️ Project Progress

### ✅ Phase 1: Data Preparation (COMPLETE)
- [x] Download and process Flickr8k dataset
- [x] Create proper train/validation/test splits
- [x] Generate clean JSONL format files
- [x] Validate image integrity and caption quality
- [x] Create comprehensive documentation

### ✅ Phase 2: Zero-Shot Baseline (COMPLETE)
- [x] Implement OpenCLIP image and text encoding
- [x] Compute baseline retrieval metrics (Recall@1/5/10)
- [x] Save normalized embeddings for efficient reuse
- [x] Achieve 50.7% text→image and 70.0% image→text Recall@1

### ✅ Phase 3: Efficient Indexing (COMPLETE)
- [x] Build FAISS HNSW index for scalable search
- [x] Optimize index parameters (M=16, efConstruction=200, efSearch=100)
- [x] Achieve sub-millisecond query speeds with zero quality loss
- [x] Create comprehensive latency vs accuracy analysis

### 🚧 Phase 4: Model Fine-Tuning (NEXT)
- [ ] Implement margin ranking loss training
- [ ] Add in-batch hard negative mining
- [ ] Conduct ablation studies on key hyperparameters
- [ ] Target: Improve text→image R@1 from 50.7% to 60%+

### 🚧 Phase 5: Production API
- [ ] Build FastAPI service with retrieval endpoints
- [ ] Add request validation and response caching
- [ ] Create Docker deployment configuration

## 📁 Project Structure

```
├── data/                          # Dataset files
│   ├── train.jsonl               # Training data (28,315 entries)
│   ├── val.jsonl                 # Validation data (6,065 entries)
│   ├── test.jsonl                # Test data (6,075 entries)
│   ├── images/                   # Processed image files (8,091 images)
│   └── README.md                 # Dataset documentation
├── artifacts/                    # Model artifacts and embeddings
│   ├── embeddings/              # Cached embeddings (val/test ready)
│   ├── indexes/                 # FAISS index files
│   │   ├── image_hnsw_index.faiss
│   │   ├── text_hnsw_index.faiss
│   │   └── index_metadata.json
│   ├── models/                  # Fine-tuned model checkpoints
│   └── project_state.json       # Current project state
├── results/                     # Evaluation results and metrics
│   ├── zero_shot_baseline_results.json
│   └── faiss_parameter_tuning.json
├── reports/                     # Analysis reports and visualizations
│   └── faiss_tradeoff_analysis.png
├── notebooks/                   # Development notebooks
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- OpenCLIP
- FAISS
- PIL (Pillow)

### Quick Start
1. **Clone the repository**
   ```bash
   git clone https://github.com/anshthamke87/clip-image-text-retrieval-for-captioned-media.git
   cd clip-image-text-retrieval-for-captioned-media
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision open-clip-torch faiss-cpu pillow numpy pandas matplotlib
   ```

3. **Load the processed dataset**
   ```python
   import json
   
   # Load training data
   with open('data/train.jsonl', 'r') as f:
       train_data = [json.loads(line) for line in f]
   
   # Load pre-computed embeddings
   import pickle
   with open('artifacts/embeddings/test_image_embeddings.pkl', 'rb') as f:
       image_embeddings = pickle.load(f)
   ```

4. **Use FAISS indexes for fast search**
   ```python
   import faiss
   
   # Load FAISS index
   index = faiss.read_index('artifacts/indexes/image_hnsw_index.faiss')
   
   # Fast similarity search
   scores, indices = index.search(query_embeddings, k=10)
   ```

## 📊 Technical Implementation

### Zero-Shot Baseline Details
- **Model**: OpenCLIP ViT-B/32 with OpenAI weights
- **Image Processing**: 224x224 resolution, standard CLIP preprocessing
- **Text Processing**: CLIP tokenizer with 77 token limit
- **Embedding Dimension**: 512-dimensional vectors, L2-normalized
- **Evaluation**: Cosine similarity search with proper train/val/test splits

### FAISS Indexing Details
- **Index Type**: Hierarchical Navigable Small World (HNSW)
- **Parameters**: M=16, efConstruction=200, optimized efSearch
- **Build Time**: <1 minute for 8k images
- **Memory Usage**: <50MB for complete indexes
- **Query Speed**: 0.1-0.2ms per query (sub-millisecond)
- **Quality**: Zero degradation vs exact search

### Performance Benchmarks
- **Evaluation Set**: 1,215 test images with 6,075 captions
- **Hardware**: Google Colab CPU (Phase 3 optimized for CPU)
- **Indexing Speed**: Sub-minute for dataset-scale indexes
- **Search Throughput**: >5,000 queries per second

## 🎯 Key Learning Outcomes

This project demonstrates:
- **Multimodal AI**: Working with vision-language models
- **Data Engineering**: Large-scale dataset processing and validation  
- **Information Retrieval**: Building efficient search systems
- **ML Engineering**: End-to-end model optimization and deployment
- **Production Optimization**: Scaling research models to real-world performance
- **Software Engineering**: Clean code, documentation, and version control

## 📈 Performance Analysis

### Search Quality vs Speed Trade-offs
The project includes comprehensive analysis of FAISS parameters:
- **efSearch tuning**: Optimal balance between speed and recall
- **Memory vs accuracy**: Index size optimization
- **Production readiness**: Real-time query capabilities

### Scalability Characteristics
- **Linear scaling**: Performance scales with dataset size
- **Memory efficient**: Indexes require minimal RAM overhead
- **CPU optimized**: Fast search without GPU requirements

## 📧 Contact

**Ansh Thamke**
- Email: at3841@columbia.edu
- GitHub: [@anshthamke87](https://github.com/anshthamke87)

---

*Built with ❤️ for advancing multimodal AI systems*
