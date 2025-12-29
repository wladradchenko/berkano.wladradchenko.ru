import onnx
from onnxconverter_common import float16


def export(model_path: str, output_name: str):
    # Load model
    model_fp32 = onnx.load(model_path)
    # Export to float16
    model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
    # Save
    onnx.save(model_fp16, output_name)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path model onnx")
    parser.add_argument("--output_name", type=str, default="fp16.onnx", help="Path to save fp16 mode with name model_fp16.onnx")
    args = parser.parse_args()

    export(model_path=args.model_path, output_name=args.output_name)

