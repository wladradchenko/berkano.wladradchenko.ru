import torch
import timm
# pip install onnxscript onnxruntime onnxruntime-tools
import argparse


def load_class_mapping(class_list_file):
    with open(class_list_file) as f:
        class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
    return class_index_to_class_name


def export(model_path: str, output_name: str):
    torch.serialization.add_safe_globals([argparse.Namespace])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Class and model after train.py
    class_mapping = load_class_mapping(args.class_mapping)

    # Load model
    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(class_mapping))
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Dummy input (ViT Base 224x224 RGB)
    dummy_input = torch.randn(1, 3, 518, 518)

    # Export
    torch.onnx.export(
        model,                      # Model
        dummy_input,                # Input
        output_name,           # Name
        export_params=True,         # Save weights
        opset_version=18,           # Version
        input_names=['input'],      # Name of inputs
        output_names=['output'],    # Name of outputs
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print("Finished!")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--class_mapping", type=str, required=True, help="Path to class mapping")
    parser.add_argument("--model_path", type=str, required=True, help="Path species mapping")
    parser.add_argument("--output_name", type=str, default="plant_classificator.onnx", help="Path to save onnx model with name model.onnx")
    args = parser.parse_args()

    export(model_path=args.model_path, output_name=args.output_name)

