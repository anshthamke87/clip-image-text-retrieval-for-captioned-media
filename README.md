# CLIP Image-Text Retrieval Engine for Captioned Media

A comprehensive implementation of bi-directional image-text retrieval using OpenCLIP, FAISS indexing, and fine-tuning techniques.

## 🎯 Project Overview

This project implements a complete pipeline for building a scalable image-text retrieval system that can:
- **Search images using text queries** (text → image retrieval)
- **Find relevant captions for images** (image → text retrieval)
- **Scale to large datasets** using efficient ANN indexing
- **Improve through fine-tuning** with ranking losses

## 📊 Current Results (Zero-Shot Baseline)

**OpenCLIP ViT-B/32 on Flickr8k Test Set:**

### Text → Image Retrieval:
- **Recall@1**: 50.7% (1 in 2 captions finds correct image)
- **Recall@5**: 76.9% (3 in 4 captions find correct image in top-5)
- **Recall@10**: 85.8% (excellent top-10 performance)

### Image → Text Retrieval:
- **Recall@1**: 70.0% (7 in 10 images find relevant caption)
- **Recall@5**: 89.1% (9 in 10 images find good captions)
- **Recall@10**: 94.9% (outstanding coverage)

*These are strong baseline results that demonstrate the effectiveness of pre-trained CLIP models!*

## 🏗️ Architecture

- **Base Model**: OpenCLIP ViT-B/32 (pre-trained vision-language model)
- **Indexing**: FAISS HNSW for approximate nearest neighbor search (next phase)
- **Training**: Margin ranking loss with in-batch hard negatives (planned)
- **Dataset**: Flickr8k (8,091 images, 40,455 captions)
- **API**: FastAPI service for real-time retrieval (planned)

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

### 🚧 Phase 3: Efficient Indexing (NEXT)
- [ ] Build FAISS HNSW index for scalable search
- [ ] Optimize index parameters (M, efConstruction, efSearch)
- [ ] Benchmark retrieval latency vs accuracy trade-offs

### 🚧 Phase 4: Model Fine-Tuning
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
│   ├── models/                  # Fine-tuned model checkpoints
│   ├── indexes/                 # FAISS index files
│   └── project_state.json       # Current project state
├── results/                     # Evaluation results and metrics
│   └── zero_shot_baseline_results.json
├── notebooks/                   # Development notebooks
├── reports/                     # Analysis reports and visualizations
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
   pip install torch torchvision open-clip-torch faiss-cpu pillow numpy pandas
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

## 📊 Technical Implementation

### Zero-Shot Baseline Details
- **Model**: OpenCLIP ViT-B/32 with OpenAI weights
- **Image Processing**: 224x224 resolution, standard CLIP preprocessing
- **Text Processing**: CLIP tokenizer with 77 token limit
- **Embedding Dimension**: 512-dimensional vectors, L2-normalized
- **Evaluation**: Cosine similarity search with proper train/val/test splits

### Performance Benchmarks
- **Evaluation Set**: 1,215 test images with 6,075 captions
- **Hardware**: Google Colab T4 GPU
- **Encoding Speed**: ~8k images in 15-20 minutes
- **Memory Usage**: 512MB for embeddings storage

## 🎯 Key Learning Outcomes

This project demonstrates:
- **Multimodal AI**: Working with vision-language models
- **Data Engineering**: Large-scale dataset processing and validation  
- **Information Retrieval**: Building efficient search systems
- **ML Engineering**: End-to-end model evaluation and optimization
- **Software Engineering**: Clean code, documentation, and version control

## 📧 Contact

**Ansh Thamke**
- Email: at3841@columbia.edu
- GitHub: [@anshthamke87](https://github.com/anshthamke87)

---

*Built with ❤️ for advancing multimodal AI systems*
