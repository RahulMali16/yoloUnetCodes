import React, { useState, useRef } from 'react';
import * as ort from 'onnxruntime-web';

const App = () => {
  const [yoloModel, setYoloModel] = useState(null);
  const canvasRef = useRef(null);

  const loadModel = async () => {
    if (!yoloModel) {
      const model = await ort.InferenceSession.create('/yolo.onnx');
      setYoloModel(model);
      return model;
    }
    return yoloModel;
  };

  const imageToTensor = async (img) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    canvas.width = 512;
    canvas.height = 512;
    ctx.drawImage(img, 0, 0, 512, 512);
    const imageData = ctx.getImageData(0, 0, 512, 512).data;

    const input = new Float32Array(3 * 512 * 512);
    for (let i = 0; i < 512 * 512; i++) {
      input[i] = imageData[i * 4] / 255.0;               // R
      input[i + 512 * 512] = imageData[i * 4 + 1] / 255.0; // G
      input[i + 2 * 512 * 512] = imageData[i * 4 + 2] / 255.0; // B
    }

    return input;
  };

  const parseYOLOOutput = (output, threshold = 0.5) => {
    const allBoxes = [];
    const outputArray = Array.isArray(output) ? output : [output]; // handle multiple outputs
    outputArray.forEach(out => {
      const data = out.data;
      const numBoxes = out.dims[1]; // assuming format [1, num, 6]
      for (let i = 0; i < numBoxes; i++) {
        const offset = i * 6;
        const [x, y, w, h, conf, cls] = data.slice(offset, offset + 6);
        if (conf > threshold) {
          allBoxes.push({ x, y, w, h, conf, cls });
        }
      }
    });
    return allBoxes;
  };

  const drawBoxes = (boxes, ctx) => {
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    boxes.forEach((box) => {
      const x = (box.x - box.w / 2) * 512;
      const y = (box.y - box.h / 2) * 512;
      const w = box.w * 512;
      const h = box.h * 512;
      ctx.strokeRect(x, y, w, h);
      ctx.font = "16px Arial";
      ctx.fillStyle = "red";
      ctx.fillText(`Conf: ${box.conf.toFixed(2)}`, x, y > 10 ? y - 5 : y + 15);
    });
  };

  const handleFile = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const img = new Image();
    img.onload = async () => {
      const model = await loadModel();
      const tensorData = await imageToTensor(img);
      const tensor = new ort.Tensor('float32', tensorData, [1, 3, 512, 512]);

      // prepare input (update key based on your model)
      const feeds = {};
      const inputName = model.inputNames[0];
      feeds[inputName] = tensor;

      const outputMap = await model.run(feeds);

      // handle all outputs (could be >1)
      const outputTensors = Object.values(outputMap);
      const boxes = parseYOLOOutput(outputTensors);

      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, 512, 512);
      ctx.drawImage(img, 0, 0, 512, 512);
      drawBoxes(boxes, ctx);
    };
    img.src = URL.createObjectURL(file);
  };

  return (
    <div>
      <h2>YOLO Object Detection - Multiple Boxes</h2>
      <input type="file" accept="image/*" onChange={handleFile} />
      <canvas ref={canvasRef} width={512} height={512} style={{ border: '1px solid white' }} />
    </div>
  );
};

export default App;
