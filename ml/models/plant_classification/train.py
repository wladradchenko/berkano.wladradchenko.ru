"""
Script for retraining the vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all model
on new datasets with garden plants.

Run:
    python train.py \
        --data_dir /path/to/your/data \
        --pretrained_path /path/to/model_best.pth.tar or None \
        --class_mapping /path/to/class_mapping.txt\
        --output_dir ./outputs \
        --epochs 50 \
        --batch_size 32 \
        --lr 1e-5
"""

import argparse
import os
import yaml
import torch
from types import SimpleNamespace
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
import timm.data
from timm.utils import model_ema
from timm.utils import accuracy, AverageMeter
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import shutil


def load_class_mapping(class_list_file):
    """Loads the mapping of class indices into species_id"""
    with open(class_list_file) as f:
        class_index_to_species_id = {i: line.strip() for i, line in enumerate(f)}
    return class_index_to_species_id


def load_species_mapping(species_map_file):
    """Loads the mapping species_id to species names"""
    df = pd.read_csv(species_map_file, sep=';', quoting=1, dtype={'species_id': str})
    df = df.set_index('species_id')
    return df['species'].to_dict()


class ClassificationDataset(Dataset):
    """
    A dataset for training a plant classification model.

    Supports two data formats:
    1. Folder structure: data_dir/train/class_name/image.jpg
    2. CSV file: data_dir/train.csv with 'image_path' and 'label' columns
    """

    def __init__(self, data_dir, split='train', class_to_idx=None, transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.samples = []

        # Checking the data format
        csv_path = self.data_dir / f'{split}.csv'
        folder_path = self.data_dir / split

        if csv_path.exists():
            # CSV
            df = pd.read_csv(csv_path)
            assert 'image_path' in df.columns and 'label' in df.columns, "CSV must contain columns 'image_path' and 'label'"

            # Create class_to_idx if not provided
            if class_to_idx is None:
                unique_classes = sorted(df['label'].unique())
                class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

            for _, row in df.iterrows():
                img_path = Path(row['image_path'])
                if not img_path.is_absolute():
                    img_path = self.data_dir / img_path
                label_str = str(row['label'])
                label_idx = class_to_idx[label_str]
                if img_path.exists():
                    self.samples.append((str(img_path), label_idx))

        elif folder_path.exists():
            # Folder format
            if class_to_idx is None:
                classes = sorted([d.name for d in folder_path.iterdir() if d.is_dir()])
                class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

            for class_name in class_to_idx.keys():
                class_dir = folder_path / class_name
                if class_dir.exists():
                    img_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
                    for img_file in img_files:
                        self.samples.append((str(img_file), class_to_idx[class_name]))
        else:
            raise ValueError(f"No CSV file found ({csv_path}), no folder ({folder_path})")

        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

        print(f"Uploaded {len(self.samples)} samples {split}")
        print(f"Number of classes: {len(self.class_to_idx)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Returning a black image as a fallback
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        return image, label_idx


def create_model(num_classes, pretrained_path=None, device='cuda'):
    """
    Creates a model and loads pretrained weights.
    If num_classes differs from the original, replaces the classifier.
    """
    # Create a model with the required number of classes
    model = timm.create_model(
        'vit_base_patch14_reg4_dinov2.lvd142m',
        pretrained=False,
        num_classes=num_classes
    )

    # Loading pre-trained weights
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pre-trained weights from {pretrained_path}")
        torch.serialization.add_safe_globals([argparse.Namespace])
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

        # Process different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Remove the 'module.' prefix if present (for models saved with DataParallel)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        num_old_classes = state_dict['head.weight'].shape[0]
        print(f"Number old classes: {num_old_classes}. Number new classes: {num_classes}")

        # Create a new head
        old_head = model.head
        in_features = old_head.in_features
        new_head = nn.Linear(in_features, num_classes)

        # Copy the old weights from state_dict
        with torch.no_grad():
            new_head.weight[:num_old_classes] = state_dict['head.weight']
            new_head.bias[:num_old_classes] = state_dict['head.bias']

        model.head = new_head
        # Load the remaining weights (except the head)
        state_dict_filtered = {k: v for k, v in state_dict.items() if 'head' not in k}
        model.load_state_dict(state_dict_filtered, strict=False)
        print("Weights loaded successfully")
    else:
        print("Pre-trained weights not found, initializing from scratch")

    model = model.to(device)
    return model


def train_epoch(model, loader, criterion, optimizer, device, epoch, args):
    """One era of learning"""
    model.train()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Метрики
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1.item(), images.size(0))
        acc5_meter.update(acc5.item(), images.size(0))

        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc1': f'{acc1_meter.avg:.2f}%',
            'acc5': f'{acc5_meter.avg:.2f}%'
        })

        if batch_idx % args.log_interval == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(loader)}, '
                  f'Loss: {loss_meter.avg:.4f}, Acc1: {acc1_meter.avg:.2f}%, Acc5: {acc5_meter.avg:.2f}%')

    return loss_meter.avg, acc1_meter.avg, acc5_meter.avg


def validate(model, loader, criterion, device):
    """Model validation"""
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Validation'):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            loss_meter.update(loss.item(), images.size(0))
            acc1_meter.update(acc1.item(), images.size(0))
            acc5_meter.update(acc5.item(), images.size(0))

    return loss_meter.avg, acc1_meter.avg, acc5_meter.avg


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"The device is in use: {device}")

    # Loading class mappings
    if args.class_mapping:
        class_mapping = load_class_mapping(args.class_mapping)
        num_classes = len(class_mapping)
        print(f"Loaded {num_classes} classes from class_mapping.txt")
    else:
        # Determine the number of classes from the data
        train_dataset_temp = ClassificationDataset(args.data_dir, split='train', transform=None)
        num_classes = len(train_dataset_temp.class_to_idx)
        class_mapping = train_dataset_temp.idx_to_class
        print(f"Detected {num_classes} classes")

    # Create model
    model = create_model(num_classes, args.pretrained_path, device)

    # Obtaining data configuration for the model
    data_config = timm.data.resolve_model_data_config(model)
    print(f"Model data configuration: {data_config}")

    # Creating transformations
    train_transform = timm.data.create_transform(
        **data_config,
        is_training=True,
        scale=(0.08, 1.0),
        ratio=(0.75, 1.3333333333333333),
        hflip=0.5,
        color_jitter=0.4
    )

    val_transform = timm.data.create_transform(
        **data_config,
        is_training=False
    )

    # Create dataset
    train_dataset = ClassificationDataset(
        args.data_dir,
        split='train',
        class_to_idx=None,  # It will be created automatically
        transform=train_transform
    )

    val_dataset = ClassificationDataset(
        args.data_dir,
        split='val',
        class_to_idx=train_dataset.class_to_idx,  # Use the same mapping
        transform=val_transform
    )

    # Preserving class mapping
    class_mapping_file = os.path.join(args.output_dir, 'class_mapping.txt')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(class_mapping_file, 'w') as f:
        if args.pretrained_path:
            for idx in sorted(class_mapping.keys()):
                f.write(f"{class_mapping[idx]}\n")
        else:
            for idx in sorted(train_dataset.idx_to_class.keys()):
                f.write(f"{train_dataset.idx_to_class[idx]}\n")
    print(f"Class mapping saved in {class_mapping_file}")

    # Creating data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # Loss function
    if args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optim_params = SimpleNamespace()
    optim_params.weight_decay = args.weight_decay
    optim_params.lr = args.lr
    optim_params.opt = args.opt  # 'lookahead_adam' to use `lookahead`
    optim_params.momentum = 0.9
    optimizer = create_optimizer(optim_params, model)

    # Training planner
    sched_params = SimpleNamespace()
    sched_params.sched = args.sched  # 'cosine', 'step', 'exponential', ...
    sched_params.num_epochs = args.epochs
    sched_params.warmup_epochs = args.warmup_epochs
    sched_params.warmup_lr = args.warmup_lr
    sched_params.min_lr = args.min_lr
    lr_scheduler, _ = create_scheduler(sched_params, optimizer, updates_per_epoch=0)

    # EMA (Exponential Moving Average)
    model_ema_model = None
    if args.model_ema:
        model_ema_model = model_ema.ModelEma(
            model,
            decay=args.model_ema_decay,
            device=device
        )

    # Train
    best_acc1 = 0.0
    history = []

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc1, train_acc5 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args
        )

        # Valid
        val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion, device)

        # Update learning rate
        lr_scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        # Update EMA
        if model_ema_model is not None:
            model_ema_model.update(model)

        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc1': train_acc1,
            'train_acc5': train_acc5,
            'val_loss': val_loss,
            'val_acc1': val_acc1,
            'val_acc5': val_acc5,
            'lr': current_lr
        })

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc1: {train_acc1:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc1: {val_acc1:.2f}%, LR: {current_lr:.6f}")

        # Save best model
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
            'class_to_idx': train_dataset.class_to_idx,
        }

        if model_ema_model is not None:
            checkpoint['ema_state_dict'] = model_ema_model.state_dict()

        # Save last checkpoint
        last_path = os.path.join(args.output_dir, 'last.pth.tar')
        torch.save(checkpoint, last_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(args.output_dir, 'model_best.pth.tar')
            torch.save(checkpoint, best_path)
            print(f"The best model with accuracy is saved {best_acc1:.2f}%")

    # Saving learning history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(args.output_dir, 'training_history.csv'), index=False)
    print(f"The training history is saved in {args.os.path.join(args.output_dir, 'training_history.csv')}")

    print(f"Training complete! Better accuracy: {best_acc1:.2f}%")


if __name__ == '__main__':
    # https://huggingface.co/datasets/Project-AgML/iNatAg
    parser = argparse.ArgumentParser(description='Дообучение модели классификации растений')

    # Пути
    parser.add_argument('--data_dir', type=str, required=True, help='Путь к директории с данными (должна содержать train/ и val/ папки или train.csv и val.csv)')
    parser.add_argument('--pretrained_path', type=str, default=None, required=False, help='Путь к предобученной модели (model_best.pth.tar)')
    parser.add_argument('--class_mapping', type=str, default=None, help='Путь к файлу class_mapping.txt (опционально, если не указан, будет создан автоматически)')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Директория для сохранения результатов')

    # Гиперпараметры обучения
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=32, help='Размер батча')
    parser.add_argument('--lr', type=float, default=1e-5,  help='Начальный learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--opt', type=str, default='adam',  help='Оптимизатор (adam, sgd, etc.)')
    parser.add_argument('--sched', type=str, default='cosine', help='Планировщик learning rate (cosine, step, etc.)')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Количество эпох warmup')
    parser.add_argument('--warmup_lr', type=float, default=1e-5, help='Learning rate для warmup')
    parser.add_argument('--min_lr', type=float, default=0.0, help='Минимальный learning rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')

    # EMA
    parser.add_argument('--model_ema', action='store_true', help='Использовать Exponential Moving Average')
    parser.add_argument('--model_ema_decay', type=float, default=0.9998, help='Decay для EMA')

    # Другие параметры
    parser.add_argument('--workers', type=int, default=4, help='Количество worker процессов для загрузки данных')
    parser.add_argument('--device', type=str, default='cuda', help='Устройство для обучения (cuda или cpu)')
    parser.add_argument('--log_interval', type=int, default=100, help='Интервал логирования')

    args = parser.parse_args()

    main(args)
