# Inference (PTH / ONNX)

This script provides inference for the LeafNet model, predicting plant age and leaf count from a single image. The script supports both PyTorch `.pth` models and ONNX `.onnx` models optimized for mobile devices.

The model uses a MobileNet backbone with two regression heads for leaf count and age estimation. Input images should be RGB and resized to 224x224 pixels. Outputs are returned as a dictionary containing `leaf_count` and `age` values.

## Setup

Clone the repository containing the model:

```bash
git clone https://huggingface.co/wladradchenko/berkano.wladradchenko.ru
```

Create a `checkpoints` directory in the current working directory and place the ONNX model there:

```bash
mkdir checkpoints
cp berkan.wladradchenko.ru/model/plant_analysis.onnx checkpoints/
```

Install required dependencies:

```bash
pip install torch torchvision onnxruntime pillow numpy
```

## Usage

Run the inference script with the path to the model and an image file or URL:

```bash
python inference.py --model_path checkpoints/plant_analysis.onnx --image example/radish_p2_d13_L2_300.png
```

The output will be a JSON-like dictionary:

```json
{
    "leaf_count": 6.18,
    "age": 13.16
}
```

If using a `.pth` model instead of ONNX, the same script can be used by replacing the model path:

```bash
python inference.py --model_path checkpoints/plant_analysis.pth --image example/radish_p2_d13_L2_300.png
```

The script automatically detects whether to use PyTorch or ONNX runtime and will run on GPU if available.
