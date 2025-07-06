import React, { useRef, useState } from 'react';
import * as ort from 'onnxruntime-web';

const App = () => {
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);

  const loadModel = async () => {
    if (!model) {
      const session = await ort.InferenceSession.create('./unet.onnx');
      setModel(session);
    }
  };

  const prepareInput = (ctx) => {
    const imageData = ctx.getImageData(0, 0, 16, 16).data;
    const gray = new Float32Array(16 * 16);
    for (let i = 0; i < 256; i++) {
      const r = imageData[i * 4];
      const g = imageData[i * 4 + 1];
      const b = imageData[i * 4 + 2];
      gray[i] = (r + g + b) / (3 * 255); // normalize
    }
    return gray;
  };

  const drawOutputMask = (ctx, mask) => {
    const output = ctx.createImageData(16, 16);
    for (let i = 0; i < 256; i++) {
      const active = mask[i] > 0.5 ? 255 : 0;
      output.data[i * 4 + 0] = 255;
      output.data[i * 4 + 1] = 0;
      output.data[i * 4 + 2] = 0;
      output.data[i * 4 + 3] = active;
    }
    ctx.putImageData(output, 0, 0);
  };

  const handleImage = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const img = new Image();
    img.onload = async () => {
      await loadModel();

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, 16, 16);

      const inputArray = prepareInput(ctx);
      const tensor = new ort.Tensor('float32', inputArray, [1, 16, 16, 1]);
      const inputName = model.inputNames[0];
      const feeds = { [inputName]: tensor };

      const results = await model.run(feeds);
      const mask = Object.values(results)[0].data;

      drawOutputMask(ctx, mask);
    };
    img.src = URL.createObjectURL(file);
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Run ONNX UNet Model in React</h2>
      <input type="file" accept="image/*" onChange={handleImage} />
      <canvas
        ref={canvasRef}
        width={16}
        height={16}
        style={{ border: '1px solid white', imageRendering: 'pixelated', marginTop: 10 }}
      />
    </div>
  );
};

export default App;
