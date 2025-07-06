import React, { useRef, useState } from 'react';
import * as ort from 'onnxruntime-web';

const UNetOnlyApp = () => {
  const canvasRef = useRef(null);
  const [unetModel, setUnetModel] = useState(null);

  const loadModel = async () => {
    if (!unetModel) {
      const model = await ort.InferenceSession.create('/unet.onnx');
      setUnetModel(model);
    }
  };

  const prepareGrayscaleInput = (ctx) => {
    const imageData = ctx.getImageData(0, 0, 16, 16).data;
    const gray = new Float32Array(16 * 16);
    for (let i = 0; i < 256; i++) {
      const r = imageData[i * 4];
      const g = imageData[i * 4 + 1];
      const b = imageData[i * 4 + 2];
      gray[i] = (r + g + b) / (3 * 255); // normalized grayscale
    }
    return gray;
  };

  const drawSegmentation = (ctx, maskData) => {
    const maskImage = ctx.createImageData(16, 16);
    for (let i = 0; i < 256; i++) {
      const active = maskData[i] > 0.5 ? 255 : 0;
      maskImage.data[i * 4 + 0] = 255; // red
      maskImage.data[i * 4 + 1] = 0;
      maskImage.data[i * 4 + 2] = 0;
      maskImage.data[i * 4 + 3] = active;
    }
    ctx.putImageData(maskImage, 0, 0);
  };

  const handleImage = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const img = new Image();
    img.onload = async () => {
      await loadModel();

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, 16, 16);
      ctx.drawImage(img, 0, 0, 16, 16); // resize to 16x16

      const input = prepareGrayscaleInput(ctx);
      const tensor = new ort.Tensor('float32', input, [1, 16, 16, 1]);
      const feeds = { [unetModel.inputNames[0]]: tensor };

      const results = await unetModel.run(feeds);
      const mask = Object.values(results)[0].data;

      drawSegmentation(ctx, mask);
    };
    img.src = URL.createObjectURL(file);
  };

  return (
    <div>
      <h2>U-Net Segmentation (16×16 Input)</h2>
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

export default UNetOnlyApp;
