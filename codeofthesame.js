import React, { useRef, useState } from "react";
import * as ort from "onnxruntime-web";

const App = () => {
  const canvasRef = useRef(null);
  const [outputUrl, setOutputUrl] = useState(null);

  // Preprocess the input image to match ONNX input dimensions and format
  const preprocess = async (imageBitmap) => {
    const width = 512;
    const height = 512;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(imageBitmap, 0, 0, width, height);

    const imageData = ctx.getImageData(0, 0, width, height).data;

    const floatArray = new Float32Array(width * height * 3);
    for (let i = 0; i < width * height; i++) {
      floatArray[i] = imageData[i * 4] / 255;                        // R
      floatArray[i + width * height] = imageData[i * 4 + 1] / 255;  // G
      floatArray[i + 2 * width * height] = imageData[i * 4 + 2] / 255; // B
    }

    return new ort.Tensor("float32", floatArray, [1, 3, height, width]);
  };

  // Postprocess the output mask into an image
  const postprocess = (maskData, width, height) => {
    const imageData = new Uint8ClampedArray(width * height * 4);

    for (let i = 0; i < width * height; i++) {
      const pixel = maskData[i] > 0.5 ? 255 : 0;
      imageData[i * 4] = pixel;     // Red
      imageData[i * 4 + 1] = 0;     // Green
      imageData[i * 4 + 2] = 0;     // Blue
      imageData[i * 4 + 3] = 255;   // Fully opaque
    }

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    const imgData = new ImageData(imageData, width, height);
    ctx.putImageData(imgData, 0, 0);

    return canvas.toDataURL();
  };

  // Main handler for image input and inference
  const handleImage = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const imageBitmap = await createImageBitmap(file);
    const inputTensor = await preprocess(imageBitmap);

    // Load the ONNX model
    const session = await ort.InferenceSession.create("/unet_model.onnx");

    // Adjust input/output names based on your model
    const feeds = { input: inputTensor }; // 'input' should match your ONNX model input name
    const results = await session.run(feeds);

    const outputName = session.outputNames[0]; // gets first output key (like 'output' or 'mask')
    const mask = results[outputName].data;

    const output = postprocess(mask, 512, 512);
    setOutputUrl(output);
  };

  return (
    <div className="App" style={{ textAlign: "center", marginTop: "30px" }}>
      <h2>U-Net Image Segmentation</h2>
      <input type="file" accept="image/*" onChange={handleImage} />
      <br /><br />
      <canvas ref={canvasRef} style={{ display: "none" }} />
      {outputUrl && (
        <div>
          <h3>Segmented Output:</h3>
          <img src={outputUrl} alt="Segmented Mask" style={{ border: "1px solid #ccc" }} />
        </div>
      )}
    </div>
  );
};

export default App;
