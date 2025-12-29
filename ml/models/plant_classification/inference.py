import onnxruntime as ort
from io import BytesIO
from PIL import Image
import timm
import torch
from urllib.request import urlopen
import pandas as pd
from torchvision import transforms


def load_class_mapping(class_list_file):
    with open(class_list_file) as f:
        class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
    return class_index_to_class_name


def load_species_mapping(species_map_file):
    df = pd.read_csv(species_map_file, sep=';', quoting=1, dtype={'species_id': str})
    df = df.set_index('species_id')
    return df['species'].to_dict()


class PlantClassification:
    def __init__(self, model_path: str, class_mapping: dict, species_mapping: list, device: str = "cuda"):
        self.device = device
        # Load classes
        self.class_mapping = class_mapping
        self.num_classes = len(class_mapping)
        # Load names
        self.species_mapping = species_mapping
        # Load models
        self.is_onnx = True if model_path.endswith("onnx") else False
        self.model = self.load_model(model_path, device)
        self.transform = transforms.Compose([
            transforms.Resize(
                (518, 518),
                interpolation=Image.BICUBIC
            ),
            transforms.ToTensor(),  # float32, /255, CHW
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __call__(self, image_input):
        if self.is_onnx:
            items = self.predict_onnx(image_input)
        else:
            items = self.predict(image_input)
        return {self.species_mapping[self.class_mapping[idx]]: float(proba) for idx, proba in items}

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
            model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False,  num_classes=self.num_classes, checkpoint_path=model_path)
            model = model.to(device)
            model = model.eval()
            return model

    @staticmethod
    def load_image(image_input):
        if isinstance(image_input, bytes):
            image_input = BytesIO(image_input)
        elif 'https://' in image_input or 'http://' in image_input:
            image_input = urlopen(image_input)
        return image_input

    def predict_onnx(self, image_input, top_k: int = 5) -> dict:
        img_tensor = self.transform(Image.open(self.load_image(image_input)).convert("RGB")).unsqueeze(0)
        img_tensor = img_tensor.to(torch.float32)
        image_np = img_tensor.cpu().numpy()  # ONNXRuntime accepts numpy
        # Run it through the model
        outputs = self.model.run(None, {"input": image_np})
        output = outputs[0]  # numpy array shape: (1, EMBEDDING_DIM)
        indices = output[0].argsort()[-top_k:][::-1]
        probs = output[0][indices]
        return zip(indices, probs)

    def predict(self, image_input) -> dict:
        image = self.transform(Image.open(self.load_image(image_input)).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
        probs, indices = torch.topk(outputs.softmax(dim=1) * 100, k=5)
        probs = probs.cpu().detach().numpy()
        indices = indices.cpu().detach().numpy()
        return zip(indices[0], probs[0])


def main(model_path: str, class_mapping: str, species_mapping: str, image: str):
    class_mapping = load_class_mapping(class_mapping)
    species_mapping = load_species_mapping(species_mapping)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    classificator = PlantClassification(model_path, class_mapping, species_mapping, device)
    print(classificator(image))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--class_mapping", type=str, required=True, help="Path to class mapping")
    parser.add_argument("--species_mapping", type=str, required=True, help="Path species mapping")
    parser.add_argument("--model_path", type=str, required=True, help="Path model pth.tar or onnx")
    args = parser.parse_args()

    main(model_path=args.model_path, image=args.image, class_mapping=args.class_mapping, species_mapping=args.species_mapping)
