# Plant Disease Detection

This script allows you to detect plant diseases by comparing image embeddings with a precomputed database of disease embeddings. It uses a visual model that converts images into embeddings and finds the most similar disease embeddings to determine the disease name.

The model is based on **SCOLD: A Vision-Language Foundation Model for Leaf Disease Identification**, optimized for mobile devices. Text comparison weights (BERT) have been removed and the architecture has been restructured for efficient inference.

---

## Requirements

Install the following dependencies:

```bash
pip install torch torchvision onnxruntime pillow numpy scipy
```

---

## Model and Assets

Clone the repository containing the model and asset files:

```bash
git clone https://huggingface.co/wladradchenko/berkano.wladradchenko.ru
```

Then create a folder `checkpoints` in the same directory as `interface.py` and place the following files inside it:

```
checkpoints/
├── disease_detection.onnx   # Model file
├── embeddings.bin           # Precomputed disease embeddings
└── captions.json            # Disease names corresponding to embeddings
```

---

## Usage

Run the disease detection script:

```bash
python interface.py \
    --model_path checkpoints/disease_detection.onnx \
    --image path/to/your/leaf_image.jpg \
    --top_k 10
```

### Arguments

* `--model_path` – Path to the model file (`.onnx` or `.pth`).
* `--image` – Path to a local image or a URL.
* `--top_k` – Number of top predicted diseases to return (default: 10).

---

## How It Works

1. The model processes the input image to generate an embedding vector.
2. The embedding is compared to all vectors in `embeddings.bin` using cosine similarity.
3. The top-k most similar embeddings are selected.
4. The disease names corresponding to these embeddings are returned using `captions.json`.

### Example Output

```python
['Powdery mildew', 'Leaf spot', 'Rust', 'Bacterial blight', 'Anthracnose']
```

The output is a list of the most probable diseases detected in the plant leaf image.

---

## Notes

* The script supports both ONNX and PyTorch models.
* For ONNX, inference is faster and supports CPU or GPU.
* Ensure images are in RGB format. The script will automatically handle resizing and normalization.
* The model is optimized for mobile and desktop inference.

<!-- CONTACT -->
## Contact

Owner: [Wladislav Radchenko](https://github.com/wladradchenko/)

Email: [i@wladradchenko.ru](i@wladradchenko.ru)

Project on GitHub: [https://github.com/wladradchenko/berkano.wladradchenko.ru](https://github.com/wladradchenko/berkano.wladradchenko.ru)

<p align="right">(<a href="#top">to top</a>)</p>
