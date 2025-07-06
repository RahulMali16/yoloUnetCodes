import React, { useState, useRef } from 'react';
import * as ort from 'onnxruntime-web';

const App = () => {
  const [yoloModel, setYoloModel] = useState(null);
  const [unetModel, setUnetModel] = useState(null);
  const [boxes, setBoxes] = useState([]);
  const canvasRef = useRef(null);

  const loadModels = async () => {
    if (!yoloModel) {
      const yolo = await ort.InferenceSession.create('/yolo.onnx');
      setYoloModel(yolo);
    }
    if (!unetModel) {
      const unet = await ort.InferenceSession.create('/unet.onnx');
      setUnetModel(unet);
    }
  };

  const imageToTensor = (img) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    canvas.width = 512;
    canvas.height = 512;
    ctx.drawImage(img, 0, 0, 512, 512);
    const imageData = ctx.getImageData(0, 0, 512, 512).data;

    const input = new Float32Array(3 * 512 * 512);
    for (let i = 0; i < 512 * 512; i++) {
      input[i] = imageData[i * 4] / 255.0;
      input[i + 512 * 512] = imageData[i * 4 + 1] / 255.0;
      input[i + 2 * 512 * 512] = imageData[i * 4 + 2] / 255.0;
    }

    return input;
  };

  const parseYOLOOutput = (output, threshold = 0.5) => {
    const boxes = [];
    const outputArray = Array.isArray(output) ? output : [output];
    outputArray.forEach(out => {
      const data = out.data;
      const numBoxes = out.dims[1];
      for (let i = 0; i < numBoxes; i++) {
        const offset = i * 6;
        const [x, y, w, h, conf, cls] = data.slice(offset, offset + 6);
        if (conf > threshold) {
          boxes.push({ x, y, w, h, conf, cls });
        }
      }
    });
    return boxes;
  };

  const cropAndResizeBox = (ctx, box) => {
    const [x, y, w, h] = [
      (box.x - box.w / 2) * 512,
      (box.y - box.h / 2) * 512,
      box.w * 512,
      box.h * 512
    ];

    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = 16;
    cropCanvas.height = 16;
    const cropCtx = cropCanvas.getContext('2d');
    cropCtx.drawImage(ctx.canvas, x, y, w, h, 0, 0, 16, 16);
    const cropData = cropCtx.getImageData(0, 0, 16, 16).data;

    const gray = new Float32Array(16 * 16);
    for (let i = 0; i < 16 * 16; i++) {
      const r = cropData[i * 4];
      const g = cropData[i * 4 + 1];
      const b = cropData[i * 4 + 2];
      gray[i] = (r + g + b) / (3 * 255);
    }

    return gray;
  };

  const runUNet = async (inputPatch) => {
    const tensor = new ort.Tensor('float32', inputPatch, [1, 16, 16, 1]);
    const inputName = unetModel.inputNames[0];
    const result = await unetModel.run({ [inputName]: tensor });
    return Object.values(result)[0].data;
  };

  const drawYOLOBoxes = (boxes, ctx) => {
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    boxes.forEach((box) => {
      const x = (box.x - box.w / 2) * 512;
      const y = (box.y - box.h / 2) * 512;
      const w = box.w * 512;
      const h = box.h * 512;
      ctx.strokeRect(x, y, w, h);
    });
  };

  const drawUNetMaskOnInputImage = (ctx, box, maskData) => {
    const [x, y, w, h] = [
      (box.x - box.w / 2) * 512,
      (box.y - box.h / 2) * 512,
      box.w * 512,
      box.h * 512
    ];

    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = 16;
    maskCanvas.height = 16;
    const maskCtx = maskCanvas.getContext('2d');

    const maskImage = maskCtx.createImageData(16, 16);
    for (let i = 0; i < 256; i++) {
      const alpha = maskData[i] > 0.5 ? 180 : 0;
      maskImage.data[i * 4 + 0] = 255;
      maskImage.data[i * 4 + 1] = 0;
      maskImage.data[i * 4 + 2] = 0;
      maskImage.data[i * 4 + 3] = alpha;
    }

    maskCtx.putImageData(maskImage, 0, 0);
    ctx.drawImage(maskCanvas, x, y, w, h);
  };

  const handleFile = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const img = new Image();
    img.onload = async () => {
      await loadModels();

      const tensorData = imageToTensor(img);
      const tensor = new ort.Tensor('float32', tensorData, [1, 3, 512, 512]);

      const inputName = yoloModel.inputNames[0];
      const feeds = { [inputName]: tensor };
      const outputMap = await yoloModel.run(feeds);
      const outputTensors = Object.values(outputMap);
      const parsedBoxes = parseYOLOOutput(outputTensors);
      setBoxes(parsedBoxes); // ✅ store boxes to show as output below

      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, 512, 512);
      ctx.drawImage(img, 0, 0, 512, 512);
      drawYOLOBoxes(parsedBoxes, ctx);

      for (let i = 0; i < parsedBoxes.length; i++) {
        const grayPatch = cropAndResizeBox(ctx, parsedBoxes[i]);
        const mask = await runUNet(grayPatch);
        drawUNetMaskOnInputImage(ctx, parsedBoxes[i], mask);
      }
    };

    img.src = URL.createObjectURL(file);
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>YOLO + U-Net Mask on Input Image</h2>
      <input type="file" accept="image/*" onChange={handleFile} />
      <canvas ref={canvasRef} width={512} height={512} style={{ border: '1px solid white' }} />

      {boxes.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h3>YOLO Output (Bounding Boxes)</h3>
          <ul>
            {boxes.map((box, idx) => (
              <li key={idx}>
                Box {idx + 1}: x={box.x.toFixed(2)}, y={box.y.toFixed(2)}, w={box.w.toFixed(2)}, h={box.h.toFixed(2)}, conf={box.conf.toFixed(2)}, class={box.cls}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default App;
