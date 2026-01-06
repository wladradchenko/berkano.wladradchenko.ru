# Plant Classification Inference

This script allows you to classify plant species using the `plant_classification.onnx` model. The model is optimized for desktop and mobile inference and can be run using ONNX Runtime or PyTorch.

## Requirements

Make sure the following Python packages are installed:

```bash
pip install torch torchvision timm onnxruntime pillow pandas
```

## Model Download

Clone the repository containing the pre-trained models:

```bash
git clone https://huggingface.co/wladradchenko/berkano.wladradchenko.ru
```

Then create a folder named `checkpoints` in the same directory as this README and place the model file `plant_classification.onnx` inside it:

```bash
mkdir checkpoints
cp berkanowladradchenko/models/plant_classification.onnx checkpoints/
cp berkanowladradchenko/files/class_mapping.txt checkpoints/
cp berkanowladradchenko/files/species_id_to_name.txt checkpoints/
```

You also need the mapping files:

* `class_mapping.txt`
* `species_id_to_name.txt`

Place them in the same directory as your script or provide the full paths when running the script.

## File Format

**class_mapping.txt** – maps class indices to internal species IDs:

```
1355868
1355869
1355870
...
NEW_SPECIES_ID_1
NEW_SPECIES_ID_2
```

**species_id_to_name.txt** – maps species IDs to species names:

```csv
"species_id";"species"
"1355868";"Taxus baccata L."
"1355869";"Dryopteris filix-mas (L.) Schott"
"NEW_SPECIES_ID_1";"New species name"
"NEW_SPECIES_ID_2";"Another new species"
```

## Usage

Run the inference script using Python:

```bash
python inference.py \
    --model_path checkpoints/plant_classification.onnx \
    --class_mapping checkpoints/class_mapping.txt \
    --species_mapping checkpoints/species_id_to_name.txt \
    --image example/bd2d3830ac3270218ba82fd24e2290becd01317c.jpg
```

### Example Output

```json
{
  "Taxus baccata L.": 75.32,
  "Dryopteris filix-mas (L.) Schott": 12.45,
  "New species name": 8.13
}
```

The keys are species names and values are the probabilities (or scores) for the top predictions.

## Notes

* The script supports both ONNX and PyTorch formats. If you provide a `.pth.tar` checkpoint, it will automatically use PyTorch.
* The model expects images resized to 518×518 pixels and normalized using ImageNet mean and std.
* On ONNX, the top-k predictions are returned (default `k=5`).
* The device will default to `cuda` if available, otherwise `cpu`.

<!-- CONTACT -->
## Contact

Owner: [Wladislav Radchenko](https://github.com/wladradchenko/)

Email: [i@wladradchenko.ru](i@wladradchenko.ru)

Project on GitHub: [https://github.com/wladradchenko/berkano.wladradchenko.ru](https://github.com/wladradchenko/berkano.wladradchenko.ru)

<p align="right">(<a href="#top">to top</a>)</p>
