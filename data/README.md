# Flickr8k Dataset - CLIP Project (Fixed)

## Dataset Information
- **Source**: HuggingFace ariG23498/flickr8k (processed correctly)
- **Total Unique Images**: 8091
- **Total Captions**: 40455

## Processing Applied
- **Fixed grouping**: Every 5 captions grouped to 1 unique image
- **Proper splits**: Split by unique images (not by captions)
- **Image renaming**: train_000000.jpg → flickr8k_000000.jpg pattern

## Splits
- **Train**: 5663 images, 28315 captions
- **Validation**: 1213 images, 6065 captions  
- **Test**: 1215 images, 6075 captions

## File Checksums (for reproducibility)
- `train.jsonl`: a564ad8d27c90c349a3af2d47945e7b8
- `val.jsonl`: 222532afa8b2e021a0945a4edd2a31ea
- `test.jsonl`: b7d76ab1555e91c5e1ce0ad8b4633e93

## Data Format
Each JSONL file contains entries with:
- `image_path`: Relative path to image file
- `caption`: Preprocessed caption text (lowercase, normalized)
- `image_id`: Unique image identifier
- `caption_id`: Unique caption identifier
- `split`: Dataset split name

## Preprocessing Applied
- Captions converted to lowercase
- Whitespace normalized
- UTF-8 encoding ensured
- Images validated for corruption
- Images renamed for clarity (flickr8k_XXXXXX.jpg)

## Usage
```python
import json

# Load training data
with open('data/train.jsonl', 'r') as f:
    train_data = [json.loads(line) for line in f]
```

Generated on: 2025-08-15 23:46:01
