import torch
import onnxruntime as ort
from torchvision import transforms
from PIL import Image
from io import BytesIO
from urllib.request import urlopen
import numpy as np
from model import LeafNet


# ================= CLASSIFIER =================
class LeafPredictor:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.is_onnx = model_path.endswith(".onnx")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        if self.is_onnx:
            providers = ort.get_available_providers()
            provider = ["CUDAExecutionProvider"] if self.device == "cuda" and "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]
            return ort.InferenceSession(model_path, providers=provider)
        else:
            model = LeafNet()
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model

    @staticmethod
    def load_image(image_input):
        if isinstance(image_input, bytes):
            return BytesIO(image_input)
        elif isinstance(image_input, str) and (
            image_input.startswith("http://") or image_input.startswith("https://")
        ):
            return urlopen(image_input)
        return image_input

    def preprocess(self, image_input):
        image = Image.open(self.load_image(image_input)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        return tensor

    def __call__(self, image_input):
        if self.is_onnx:
            return self.predict_onnx(image_input)
        else:
            return self.predict_pth(image_input)

    # ================= PTH =================
    def predict_pth(self, image_input):
        image = self.preprocess(image_input).to(self.device)

        with torch.no_grad():
            _, count, age = self.model(image)

        return {
            "leaf_count": float(count.squeeze().cpu().item()),
            "age": float(age.squeeze().cpu().item())
        }

    # ================= ONNX =================
    def predict_onnx(self, image_input):
        image = self.preprocess(image_input)
        image_np = image.cpu().numpy()

        outputs = self.model.run(
            None,
            {"image": image_np}
        )

        # outputs: [features, count, age]
        count = outputs[1][0][0]
        age = outputs[2][0][0]

        return {
            "leaf_count": float(count),
            "age": float(age)
        }


# ================= CLI =================
def main(model_path: str, image: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = LeafPredictor(model_path, device)
    result = predictor(image)
    print(result)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="LeafNet inference (PTH / ONNX)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth or .onnx")
    parser.add_argument("--image", type=str, required=True, help="Path / URL to image")
    args = parser.parse_args()

    main(args.model_path, args.image)
