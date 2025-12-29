import json
import os
import torch
from io import BytesIO
from urllib.request import urlopen
from model import LVL
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.spatial.distance import cdist
import torch.nn.functional as F


class DiseaseDetection:
    EMBEDDINGS_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "embeddings.bin")
    LABELS_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "captions.json")

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.is_onnx = True if model_path.endswith("onnx") else False
        self.model = self.load_model(model_path, device)
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        # Templates
        self.embs = self.load_embeddings()
        self.labels = self.load_labels()

    def __call__(self, image_input, top_k) -> list:
        if self.is_onnx:
            image_emb = self.predict_onnx(image_input)
        else:
            image_emb = self.predict(image_input)
        indices = self.distance_cpu(image_emb, top_k=top_k) if self.device == "cpu" else self.distance_cuda(image_emb, top_k=top_k)
        return [self.get_label(idx) for idx in indices]

    def load_embeddings(self):
        embs_flat = np.fromfile(self.EMBEDDINGS_TEMPLATE, dtype=np.float32)
        return embs_flat.reshape(embs_flat.size // 512, 512)

    def load_labels(self) -> list:
        with open(self.LABELS_TEMPLATE, 'r', encoding='utf-8') as file:
            labels = json.load(file)
        return labels

    # Load ONNX
    def load_model(self, model_path: str, device: str = "cuda"):
        """
        Load a model, selecting a format.
        :param model_path: Path to the model.
        :param device: cuda or cpu
        :return: The loaded model.
        """
        if self.is_onnx:
            providers = ort.get_available_providers()
            if device == "cpu" or 'CUDAExecutionProvider' not in providers:
                provider = ['CPUExecutionProvider']
            else:
                provider = ['CUDAExecutionProvider']
            ort_session = ort.InferenceSession(model_path, providers=provider)
            return ort_session
        else:
            model = LVL()
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            model.to(device)
            model.eval()
            return model

    @staticmethod
    def load_image(image_input):
        if isinstance(image_input, bytes):
            image_input = BytesIO(image_input)
        elif 'https://' in image_input or 'http://' in image_input:
            image_input = urlopen(image_input)
        return image_input

    def get_label(self, idx: int):
        return self.labels[idx] if 0 <= idx < len(self.labels) else None

    @staticmethod
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def distance_cuda(self, image_emb, top_k=10):
        if not torch.is_tensor(image_emb):
            image_emb = torch.tensor(image_emb).to(self.device)
        if not torch.is_tensor(self.embs):
            self.embs = torch.tensor(self.embs).to(self.device)
        image_emb = F.normalize(image_emb, dim=-1)
        embs = F.normalize(self.embs, dim=-1)
        sims = torch.matmul(embs, image_emb.T)
        _, indices = torch.topk(sims.squeeze(1), k=top_k, largest=True)
        return indices.cpu().numpy()

    def distance_cpu(self, image_emb, top_k: int = 10):
        image_emb = self.to_numpy(image_emb)
        # Distance_matrix will be an array of shape (1, N)
        distance_matrix = cdist(image_emb.reshape(1, -1), self.embs, 'cosine')
        # Let's transform the distance matrix into a 1D array
        cosine_distances = distance_matrix.flatten()
        # Sorted in ascending order of values in the distances array
        sorted_indices = np.argsort(cosine_distances)
        top_k = min(max(0, top_k), len(sorted_indices))
        return sorted_indices[:top_k]

    def predict_onnx(self, image_input):
        image = self.transform(Image.open(self.load_image(image_input)).convert("RGB")).unsqueeze(0)
        image_np = image.cpu().numpy()  # ONNXRuntime accepts numpy
        # Run it through the model
        outputs = self.model.run(None, {"images": image_np})
        image_emb = outputs[0]  # numpy array shape: (1, EMBEDDING_DIM)
        image_emb = image_emb / np.linalg.norm(image_emb, axis=-1, keepdims=True)
        return image_emb

    def predict(self, image_input):
        image = self.transform(Image.open(self.load_image(image_input)).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_emb = self.model(image)
        return image_emb


def main(model_path: str, image: str, top_k: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    detector = DiseaseDetection(model_path, device)
    print(detector(image, top_k))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model pth or onnx")
    parser.add_argument("--image", required=True, type=str, default='https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata/test/1361687/bd2d3830ac3270218ba82fd24e2290becd01317c.jpg', help="Path to image")
    parser.add_argument("--top_k", type=int, default=10, required=False, help="Number of outputs")
    args = parser.parse_args()

    main(model_path=args.model_path, image=args.image, top_k=args.top_k)