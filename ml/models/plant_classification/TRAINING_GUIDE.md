# Plant Classification Model Training Guide

This repository provides scripts for training or fine-tuning the `vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all` model on custom plant image datasets. The model can classify images into species categories, supporting both training from scratch and transfer learning from pretrained weights.

The model is based on **Vision Transformer (ViT)** architecture with a classifier head for plant species recognition. It can be trained on datasets organized as folders or CSV files.

## Features

* Fine-tune a pretrained ViT model or train from scratch.
* Supports multiple data formats: folder structure or CSV file.
* Automatically handles class mappings.
* Supports learning rate schedulers, weight decay, label smoothing, and EMA (Exponential Moving Average).
* Outputs checkpoints and training history for analysis.

## Dataset Structure

The dataset can be organized in one of two ways:

**Folder structure:**

```
/data_dir/
    ├── train/
    │   ├── species_1/
    │   │   ├── img_001.jpg
    │   │   ├── img_002.jpg
    │   │   └── ...
    │   ├── species_2/
    │   │   ├── ...
    ├── val/
    │   ├── species_1/
    │   └── species_2/
```

**CSV file format:**

```
/data_dir/
    ├── train.csv
    ├── val.csv
```

CSV files must contain two columns: `image_path` (absolute or relative path to the image) and `label` (class name).

The script automatically creates `class_mapping.txt` if not provided, mapping each class name to an integer index.

## Requirements

Install required Python packages:

```bash
pip install torch torchvision timm pandas tqdm pillow
```

Optional packages for optimized training (if using EMA, schedulers, or advanced optimizers) are included in `timm`.

## Usage

### Command Line

Run the training script with the following parameters:

```bash
python train.py \
    --data_dir /path/to/your/data \
    --pretrained_path /path/to/model_best.pth.tar  # optional, None to train from scratch
    --class_mapping /path/to/class_mapping.txt     # optional
    --output_dir ./outputs \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-5
```

### Key Arguments

* `--data_dir`: Path to your dataset directory (must include `train/` and `val/` folders or CSV files).
* `--pretrained_path`: Path to a pretrained checkpoint (optional).
* `--class_mapping`: Optional class mapping file. Automatically generated if not provided.
* `--output_dir`: Directory to save checkpoints and training results.
* `--epochs`: Number of training epochs.
* `--batch_size`: Batch size for training.
* `--lr`: Initial learning rate.
* `--weight_decay`: Weight decay for optimizer.
* `--opt`: Optimizer type (default `adam`).
* `--sched`: Learning rate scheduler (`cosine`, `step`, etc.).
* `--warmup_epochs`: Number of warmup epochs.
* `--label_smoothing`: Label smoothing value (default 0.1).
* `--model_ema`: Use Exponential Moving Average (EMA) for smoother weights.
* `--model_ema_decay`: EMA decay factor (default 0.9998).
* `--workers`: Number of data loading workers.
* `--device`: Device for training (`cuda` or `cpu`).

### Example

```bash
python train.py \
    --data_dir ./garden_dataset \
    --pretrained_path ./outputs/model_best.pth.tar \
    --output_dir ./outputs_finetuned \
    --epochs 30 \
    --batch_size 16 \
    --lr 3e-5 \
    --opt adam \
    --sched cosine \
    --model_ema \
    --workers 8
```
### Class Mapping

The training script uses a class mapping file to convert class names to integer indices. If the file is not provided, it will be generated automatically from your dataset.

**Example `class_mapping.txt`:**

```
1355868
1355869
1355870
...
NEW_SPECIES_ID_1
NEW_SPECIES_ID_2
```

After adding new classes, update the corresponding `species_id_to_name.txt` file to map species IDs to their names:

**Example `species_id_to_name.txt`:**

```csv
"species_id";"species"
"1355868";"Taxus baccata L."
"1355869";"Dryopteris filix-mas (L.) Schott"
"NEW_SPECIES_ID_1";"New species name"
"NEW_SPECIES_ID_2";"Another new species"
```

This ensures that your training outputs can correctly translate predicted class indices back to species names.

### Outputs

* `last.pth.tar`: Last checkpoint.
* `model_best.pth.tar`: Best model based on validation top-1 accuracy.
* `class_mapping.txt`: Mapping of class names to indices.
* `training_history.csv`: Epoch-wise training and validation metrics (loss, top-1 and top-5 accuracy, learning rate).

### Notes

* The script automatically handles varying numbers of classes, replacing the classifier head if needed.
* If pretrained weights are provided, the classifier head is partially copied to support new classes.
* Images failing to load are replaced with black images to prevent training interruptions.
* Validation metrics are computed each epoch to monitor overfitting.

<!-- CONTACT -->
## Contact

Owner: [Wladislav Radchenko](https://github.com/wladradchenko/)

Email: [i@wladradchenko.ru](i@wladradchenko.ru)

Project on GitHub: [https://github.com/wladradchenko/berkano.wladradchenko.ru](https://github.com/wladradchenko/berkano.wladradchenko.ru)

<p align="right">(<a href="#top">to top</a>)</p>
