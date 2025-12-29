import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import mobilenet_v3_large
from sklearn.metrics import mean_absolute_error, r2_score
from argparse import ArgumentParser

parser = ArgumentParser(description='Дообучение модели определения возраста и кол. листов')
parser.add_argument("--dataset_root", type=str, required=True, help="Root path of dataset", default="/home/radchenko/plant/train_leaf/datasets")
parser.add_argument("--label_path", type=str, required=True, help="Path to CSV file with labels and paths to images inn dataset_root", default="/home/radchenko/plant/train_leaf/datasets/mustard_train.csv")
parser.add_argument("--output_name", type=str, default="leafnet.onnx", help="Path to save onnx model with name model.onnx")
args = parser.parse_args()

# ================= CONFIG =================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= SEED =================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ================= DATASET =================
class LeafDataset(Dataset):
    def __init__(self, root, csv_file, transform):
        self.root = root
        self.df = self._load_data(csv_file)
        self.transform = transform

    def _load_data(self, csv_file):
        df = pd.read_csv(csv_file)
        print(f"Loading {len(df)} rows from {csv_file}")
        rows = [row for _, row in df.iterrows() if os.path.exists(os.path.join(self.root, *row["filename"].split("/")))]
        print(f"Loaded {len(rows)} rows from {csv_file}")
        return pd.DataFrame(rows).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(*row["filename"].split("/"))
        img_path = os.path.join(self.root, file_path)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        age = torch.tensor(row["Age"], dtype=torch.float32)
        leaf_count = torch.tensor(row["leaf_count"], dtype=torch.float32)

        return image, leaf_count, age, row["filename"]


# ================= MODEL =================
class LeafNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_large(weights="DEFAULT")
        self.encoder = backbone.features

        # Heads
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.count_head = nn.Sequential(
            nn.Linear(960, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        self.age_head = nn.Sequential(
            nn.Linear(960, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        feat = self.encoder(x)
        pooled = self.pool(feat).flatten(1)
        count = self.count_head(pooled)
        age = self.age_head(pooled)
        return feat, count, age


# ================= TRANSFORMS =================
# transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor()
# ])

# Строки 82-85 - нужно добавить нормализацию
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet нормализация
])

# ================= DATA =================
dataset = LeafDataset(args.dataset_root, args.label_path, transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ================= TRAIN =================
model = LeafNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.L1Loss()

# Переменные для отслеживания лучшей модели
best_val_loss = float('inf')
best_val_mae_count = float('inf')
best_val_mae_age = float('inf')
best_val_r2_count = -float('inf')
best_val_r2_age = -float('inf')
best_epoch = 0

for epoch in range(EPOCHS):
    # ========== TRAINING ==========
    model.train()
    train_loss = 0
    train_pred_count, train_true_count = [], []
    train_pred_age, train_true_age = [], []

    for img, count, age, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        img, count, age = img.to(DEVICE), count.to(DEVICE), age.to(DEVICE)
        optimizer.zero_grad()
        feat, pred_count, pred_age = model(img)
        loss = loss_fn(pred_count.squeeze(), count) + loss_fn(pred_age.squeeze(), age)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Собираем предсказания для метрик
        train_pred_count.extend(pred_count.squeeze().cpu().detach().numpy())
        train_true_count.extend(count.cpu().numpy())
        train_pred_age.extend(pred_age.squeeze().cpu().detach().numpy())
        train_true_age.extend(age.cpu().numpy())

    # Вычисляем метрики для тренировки
    train_mae_count = mean_absolute_error(train_true_count, train_pred_count)
    train_mae_age = mean_absolute_error(train_true_age, train_pred_age)
    train_r2_count = r2_score(train_true_count, train_pred_count)
    train_r2_age = r2_score(train_true_age, train_pred_age)

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"  Train - Loss: {train_loss / len(train_loader):.4f} | "
          f"MAE Count: {train_mae_count:.4f} | MAE Age: {train_mae_age:.4f} | "
          f"R² Count: {train_r2_count:.4f} | R² Age: {train_r2_age:.4f}")

    # ========== VALIDATION ==========
    model.eval()
    val_loss = 0
    val_pred_count, val_true_count = [], []
    val_pred_age, val_true_age = [], []

    with torch.no_grad():
        for img, count, age, _ in val_loader:
            img, count, age = img.to(DEVICE), count.to(DEVICE), age.to(DEVICE)
            feat, pred_count, pred_age = model(img)
            val_loss += (loss_fn(pred_count.squeeze(), count) + loss_fn(pred_age.squeeze(), age)).item()

            # Собираем предсказания для метрик
            val_pred_count.extend(pred_count.squeeze().cpu().numpy())
            val_true_count.extend(count.cpu().numpy())
            val_pred_age.extend(pred_age.squeeze().cpu().numpy())
            val_true_age.extend(age.cpu().numpy())

    # Вычисляем метрики для валидации
    val_mae_count = mean_absolute_error(val_true_count, val_pred_count)
    val_mae_age = mean_absolute_error(val_true_age, val_pred_age)
    val_r2_count = r2_score(val_true_count, val_pred_count)
    val_r2_age = r2_score(val_true_age, val_pred_age)
    avg_val_loss = val_loss / len(val_loader)

    print(f"  Val   - Loss: {avg_val_loss:.4f} | "
          f"MAE Count: {val_mae_count:.4f} | MAE Age: {val_mae_age:.4f} | "
          f"R² Count: {val_r2_count:.4f} | R² Age: {val_r2_age:.4f}")

    # Сохраняем лучшую модель (по комбинированной метрике или по R²)
    # Можно выбрать критерий: лучший R² для count, или комбинированную метрику
    current_score = val_r2_count + val_r2_age  # Комбинированный R²

    if current_score > (best_val_r2_count + best_val_r2_age):
        best_val_loss = avg_val_loss
        best_val_mae_count = val_mae_count
        best_val_mae_age = val_mae_age
        best_val_r2_count = val_r2_count
        best_val_r2_age = val_r2_age
        best_epoch = epoch + 1

        # Сохраняем лучшую модель
        torch.save(model.state_dict(), "leafnet_best.pth")
        print(f"  ✓ New best model saved! (Epoch {best_epoch})")

print(f"\n{'=' * 60}")
print(f"Training completed!")
print(f"Best model at epoch {best_epoch}:")
print(f"  Val Loss: {best_val_loss:.4f}")
print(f"  Val MAE Count: {best_val_mae_count:.4f} | Val MAE Age: {best_val_mae_age:.4f}")
print(f"  Val R² Count: {best_val_r2_count:.4f} | Val R² Age: {best_val_r2_age:.4f}")
print(f"{'=' * 60}")

# Сохраняем финальную модель
torch.save(model.state_dict(), "leafnet_final.pth")
print("Final model saved as 'leafnet_final.pth'")


# ================= POST-HOC HEATMAP =================
def generate_heatmap(model, img_tensor):
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    feat, _, _ = model(img_tensor)
    heatmap = feat.mean(1).squeeze().cpu().detach().numpy()
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


# ================= GRADCAM =================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0] if grad_out[0] is not None else None

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        feat, count, age = self.model(x)
        if class_idx is None:
            score = count.sum()
        else:
            score = count[:, class_idx].sum()
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(1)
        cam = cam.relu()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = torch.nn.functional.interpolate(cam.unsqueeze(1), size=(IMG_SIZE, IMG_SIZE), mode='bilinear',
                                              align_corners=False)
        return cam.squeeze().cpu().numpy()


# ================= TEST GRADCAM =================
# model.eval()
# gradcam = GradCAM(model, target_layer=model.encoder[-1])  # последний conv слой

# Возьмем пример из валидации
# img, count, age, fname = val_ds[0]
# img_tensor = img.unsqueeze(0).to(DEVICE)
# heatmap = gradcam(img_tensor)

# plt.imshow(img.permute(1,2,0).numpy())
# plt.imshow(heatmap, cmap='jet', alpha=0.5)
# plt.title(f"Leaf Count Heatmap: {int(count.item())}, Age: {int(age.item())}")
# plt.axis('off')
# plt.show()

# ================= EXPORT ONNX =================
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
torch.onnx.export(
    model,
    dummy,
    args.output_name,
    opset_version=12,
    input_names=["image"],
    output_names=["features", "count", "age"],
    dynamic_axes={"image": {0: "batch"}}
)
