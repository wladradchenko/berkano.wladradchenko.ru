# Mobile Model for Plant Age and Leaf Count Estimation

This repository implements a convolutional neural network model based on MobileNet for estimating the age (in days) and the number of leaves of a plant from a single image. The model is trained on a dataset of radish plants at different growth stages, with images captured at various times and annotated with leaf count and age information.

The trained model is optimized for mobile inference and can be deployed both in Python-based ML projects and on mobile devices using ONNX Runtime.

## Features

The model uses a MobileNet backbone for feature extraction and separate regression heads for predicting plant age and leaf count. It supports training with configurable hyperparameters and produces numerical predictions suitable for regression tasks.

The inference is lightweight and efficient, enabling real-time predictions on mobile devices.

## Dataset Structure

The model was trained on a dataset of the format [GroMo-Plant-Growth-Modeling-with-Multiview-Images](https://arxiv.org/abs/2503.06608). The dataset consists of images of multiple plants captured over different days and categorized into multiple levels. Each image includes annotations for plant age and leaf count.

Images follow the naming convention:

```
radish_pX_dY_LZ_A.png
```

Where:

* `X` represents the plant ID (`p1, p2, ...`)
* `Y` represents the day (`d1, d2, ...`)
* `Z` represents the level (`L1, L2, L3, L4, L5`)
* `A` represents the angle or view of the image

The directory structure of the dataset is as follows:

```
/dataset/
    ├── train/
    │   ├── p1/
    │   │   ├── d1/
    │   │   │   ├── L1/
    │   │   │   │   ├── radish_p1_d1_L1_0.png
    │   │   │   │   ├── radish_p1_d1_L1_15.png
    │   │   │   │   ├── ...
    │   │   │   │   ├── radish_p1_d1_L1_345.png
    │   │   │   ├── L2/
    │   │   │   ├── ...
    │   │   ├── d2/
    │   │   ├── ...
    │   ├── p2/
    │   ├── ...
```

Each plant has images captured at different time points, with annotations for leaf count and plant age.

## Model Architecture

The model uses MobileNetV3-Large as a backbone for feature extraction. Two regression heads are attached:

* **Leaf Count Head**: predicts the number of leaves
* **Age Head**: predicts the age of the plant in days

Features are pooled using adaptive average pooling before being passed to the regression heads. The architecture is lightweight and suitable for mobile deployment.

## Training

### Requirements

```bash
pip install torch torchvision numpy pandas tqdm opencv-python
```

### Running Training

To train the model, run:

```bash
python train.py --dataset_root /path/to/dataset --label_path /path/to/labels.csv --output_name leafnet.onnx
```

Optional arguments include batch size, learning rate, number of epochs, and device selection (CPU/GPU).

The model uses L1 loss for both age and leaf count regression. Metrics such as MAE and R² are computed for evaluation.

## ONNX Export

After training, the model is exported to ONNX format for mobile inference. Both FP32 and FP16 exports are supported.

```python
torch.onnx.export(
    model,
    dummy_input,
    "leafnet.onnx",
    opset_version=18,
    input_names=["image"],
    output_names=["features", "count", "age"],
    dynamic_axes={"image": {0: "batch"}}
)
```

## Inference

The ONNX model accepts images resized to 224x224, normalized with ImageNet statistics, and formatted as `[1, 3, 224, 224]`. Inference returns:

```json
{
    "leaf_count": <predicted number of leaves>,
    "age": <predicted age in days>
}
```

Predictions can be rounded or floored as needed for display or further processing.

### Example (Python)

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("leafnet.onnx")
input_tensor = np.random.rand(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {"image": input_tensor})
leaf_count = outputs[1][0][0]
age = outputs[2][0][0]
```

### Example (React Native / Mobile)

The model can be used with `onnxruntime-react-native`, ensuring input tensors are `[1, 3, 224, 224]`, normalized and in CHW format. Output is a dictionary with `age` and `leaf_count`.

## Metrics

During training and evaluation, the following metrics are computed:

* **MAE (Leaf Count)**: mean absolute error for leaf count prediction
* **MAE (Age)**: mean absolute error for plant age prediction
* **R² (Leaf Count)**: coefficient of determination for leaf count
* **R² (Age)**: coefficient of determination for age

The model is optimized to minimize combined regression loss across both outputs.

## Usage Notes

The model is suitable for single-view plant images. It is lightweight and optimized for mobile inference, but it is trained on radish plants and may not generalize to other species without fine-tuning.

<!-- CONTACT -->
## Contact

Owner: [Wladislav Radchenko](https://github.com/wladradchenko/)

Email: [i@wladradchenko.ru](i@wladradchenko.ru)

Project on GitHub: [https://github.com/wladradchenko/berkano.wladradchenko.ru](https://github.com/wladradchenko/berkano.wladradchenko.ru)

<p align="right">(<a href="#top">to top</a>)</p>