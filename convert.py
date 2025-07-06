import torch
from unet import UNet  # ✅ Make sure UNet.py is in the same folder

# 🧠 Instantiate your model
model = UNet(in_channels=3, out_channels=1, init_features=32)
model.eval()

# 🧪 Create a dummy input tensor (adjust shape as needed)
dummy_input = torch.randn(1, 3, 128, 128)  # [batch, channels, height, width]

# 💾 Define output path
onnx_output_path = "unet.onnx"

# 🚀 Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_output_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    export_params=True,
    opset_version=11,
    do_constant_folding=True
)

print(f"✅ UNet successfully exported to {onnx_output_path}")
