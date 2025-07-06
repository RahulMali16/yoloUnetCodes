import torch
from unet_model import UNet  # or use `from unet import UNet` if your file is named unet.py

def convert_to_onnx(output_file="unet_model.onnx"):
    model = UNet(n_channels=3, n_classes=1, bilinear=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 512, 512)

    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    print(f"✅ UNet model exported to: {output_file}")

if __name__ == "__main__":
    convert_to_onnx()
