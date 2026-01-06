# Plant Analysis ML Project

This repository contains Python code and tools for plant-related machine learning tasks, including **disease detection**, **plant age and leaf count estimation**, and **plant classification**. It provides scripts for training, inference, and model optimization for both desktop and mobile usage.

---

## Project Structure

```
project_root/
├── models/                # Python code for training and inference
│   ├── disease_detection/
│   ├── plant_analysis/
│   └── plant_classification/
├── quantization/          # Scripts for ONNX model quantization
├── requirements.txt       # Python dependencies
```

### `models/`

Contains all scripts to run and train your models:

* **disease_detection/** – Scripts to generate embeddings from images and detect plant diseases by comparing with a precomputed embeddings database.
* **plant_analysis/** – Scripts to estimate plant age and leaf count using a multi-view Vision Transformer (MVVT).
* **plant_classification/** – Scripts to classify plant species based on images.

All models are compatible with PyTorch and ONNX, optimized for CPU and GPU, and suitable for mobile inference.

---

### `quantization/`

This folder contains scripts to convert ONNX models to **float16**, reducing model size and improving inference speed, especially for mobile devices. Example script:

```python
import onnx
from onnxconverter_common import float16

def export(model_path: str, output_name: str):
    # Load model
    model_fp32 = onnx.load(model_path)
    # Export to float16
    model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
    # Save
    onnx.save(model_fp16, output_name)
```

Run quantization from the command line:

```bash
python quantize_model.py --model_path path/to/model.onnx --output_name model_fp16.onnx
```

---

### `requirements.txt`

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:

* `torch` and `torchvision` – for PyTorch training and inference
* `onnx` and `onnxruntime` – for ONNX inference and model optimization
* `timm` – for Vision Transformer models
* `Pillow`, `numpy`, `scipy`, `pandas` – for data handling and image processing

---

## Models

Pretrained ONNX models can be downloaded from:

[https://huggingface.co/wladradchenko/berkano.wladradchenko.ru/](https://huggingface.co/wladradchenko/berkano.wladradchenko.ru/)

Models included:

* `disease_detection.onnx` – Plant disease detection via embeddings
* `plant_analysis.onnx` – Plant age and leaf count estimation
* `plant_classification.onnx` – Plant species classification

These models are optimized for mobile devices and can also be used in desktop ML pipelines.

---

## Usage

1. Clone the repository and install dependencies.
2. Place your ONNX models in the `models/` folder or a subdirectory.
3. Use the scripts in the respective folders to perform inference or retrain models on your datasets.
4. Optional: Run the quantization scripts to convert models to float16 for faster inference.

<!-- CONTACT -->
## Contact

Owner: [Wladislav Radchenko](https://github.com/wladradchenko/)

Email: [i@wladradchenko.ru](i@wladradchenko.ru)

Project on GitHub: [https://github.com/wladradchenko/berkano.wladradchenko.ru](https://github.com/wladradchenko/berkano.wladradchenko.ru)

<p align="right">(<a href="#top">to top</a>)</p>
