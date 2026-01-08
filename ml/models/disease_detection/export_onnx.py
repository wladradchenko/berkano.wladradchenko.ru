import torch
# pip install onnxscript onnxruntime onnxruntime-tools
from model import LVL


def export(model_path: str, output_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create a dummy input with the same dimensions as the real data.
    dummy_input = torch.randn(1, 3, 512, 512, device=device)

    # Load model
    model = LVL()
    model.to(device)
    model.eval()

    #  git clone https://huggingface.co/enalis/scold
    state_dict = torch.load(model_path, map_location=device)

    # Leave only the keys that are in the new model (only image_encoder)
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}

    # Load only image_encoder weights
    model.load_state_dict(filtered_dict, strict=False)  # <- strict=False important!

    # Export
    torch.onnx.export(
        model,                      # Model
        dummy_input,                # Input
        output_name,           # Name
        export_params=True,         # Save weights
        opset_version=18,           # Version
        do_constant_folding=True,   # Optim
        input_names=['images'],     # Name of inputs
        output_names=['image_embeddings'],  # Name of outputs
        dynamic_axes={'images': {0: 'batch_size'},  # Dynamic support batch size
                      'image_embeddings': {0: 'batch_size'}}
    )
    print("Finish!")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model pth")
    parser.add_argument("--output_name", type=str, default="disease_detector.onnx", help="Path to save onnx model with name model.onnx")
    args = parser.parse_args()

    export(model_path=args.model_path, output_name=args.output_name)
