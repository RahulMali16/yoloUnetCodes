import React, { useRef } from "react";
import * as ort from "onnxruntime-web";

const YOLO_CLASSES = ["eye1", "eye2"];

const App = () => {
  const canvasRef = useRef(null);
  const segCanvasRef = useRef(null);

  const handleUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const session = await ort.InferenceSession.create("yolov8m.onnx");
    const [input, imgWidth, imgHeight] = await prepareInput(file);
    const tensor = new ort.Tensor("float32", Float32Array.from(input), [1, 3, 512, 512]);
    const output = await session.run({ images: tensor });
    const boxes = processOutput(output.output0.data, imgWidth, imgHeight);
    drawImageAndBoxes(file, boxes);
    for (let box of boxes) runUNetOnCroppedBox(file, ...box);
  };

  const prepareInput = (file) => {
    return new Promise((resolve) => {
      const img = new Image();
      img.src = URL.createObjectURL(file);
      img.onload = () => {
        const [imgWidth, imgHeight] = [img.width, img.height];
        const canvas = document.createElement("canvas");
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, 512, 512);
        const imageData = ctx.getImageData(0, 0, 512, 512).data;

        const red = [], green = [], blue = [];
        for (let i = 0; i < imageData.length; i += 4) {
          red.push(imageData[i] / 255);
          green.push(imageData[i + 1] / 255);
          blue.push(imageData[i + 2] / 255);
        }

        const input = [...red, ...green, ...blue];
        resolve([input, imgWidth, imgHeight]);
      };
    });
  };

  const processOutput = (output, imgWidth, imgHeight) => {
    const num_classes = YOLO_CLASSES.length;
    const values_per_prediction = 4 + num_classes;
    const num_boxes = output.length / values_per_prediction;

    let boxes = [];
    for (let i = 0; i < num_boxes; i++) {
      const [classId, prob] = [...Array(num_classes).keys()]
        .map(j => [j, output[num_boxes * (j + 4) + i]])
        .reduce((a, b) => (b[1] > a[1] ? b : a), [0, 0]);

      if (prob < 0.5) continue;

      const xc = output[i];
      const yc = output[num_boxes + i];
      const w  = output[2 * num_boxes + i];
      const h  = output[3 * num_boxes + i];

      const x1 = (xc - w / 2) / 512 * imgWidth;
      const y1 = (yc - h / 2) / 512 * imgHeight;
      const x2 = (xc + w / 2) / 512 * imgWidth;
      const y2 = (yc + h / 2) / 512 * imgHeight;

      const label = YOLO_CLASSES[classId];
      boxes.push([x1, y1, x2, y2, label, prob]);
    }

    boxes.sort((a, b) => b[5] - a[5]);
    const result = [];
    while (boxes.length > 0) {
      const selected = boxes.shift();
      result.push(selected);
      boxes = boxes.filter((box) => iou(selected, box) < 0.7);
    }
    return result;
  };

  const drawImageAndBoxes = (file, boxes) => {
    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = () => {
      const canvas = canvasRef.current;
      canvas.width = img.width;
      canvas.height = img.height;

      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      ctx.lineWidth = 2;
      ctx.font = "16px Arial";

      boxes.forEach(([x1, y1, x2, y2, label, prob]) => {
        ctx.strokeStyle = "#00FF00";
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const text = `${label} (${(prob * 100).toFixed(1)}%)`;
        const textWidth = ctx.measureText(text).width;
        ctx.fillStyle = "#00FF00";
        ctx.fillRect(x1, y1 - 20, textWidth + 10, 20);
        ctx.fillStyle = "#000000";
        ctx.fillText(text, x1 + 5, y1 - 5);
      });
    };
  };

  const runUNetOnCroppedBox = async (file, x1, y1, x2, y2) => {
    const img = new Image();
    img.src = URL.createObjectURL(file);

    img.onload = async () => {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      const cropWidth = x2 - x1;
      const cropHeight = y2 - y1;

      canvas.width = cropWidth;
      canvas.height = cropHeight;
      ctx.drawImage(img, x1, y1, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);

      const smallCanvas = document.createElement("canvas");
      smallCanvas.width = 16;
      smallCanvas.height = 16;
      const smallCtx = smallCanvas.getContext("2d");
      smallCtx.drawImage(canvas, 0, 0, 16, 16);

      const imageData = smallCtx.getImageData(0, 0, 16, 16).data;
      const gray = new Float32Array(16 * 16);
      for (let i = 0; i < 16 * 16; i++) {
        const r = imageData[i * 4];
        const g = imageData[i * 4 + 1];
        const b = imageData[i * 4 + 2];
        gray[i] = (r + g + b) / (3 * 255);
      }

      const inputTensor = new ort.Tensor("float32", gray, [1, 1, 16, 16]);
      const unetSession = await ort.InferenceSession.create("unet.onnx");
      const output = await unetSession.run({ input_1: inputTensor });

      const mask = output[Object.keys(output)[0]].data;
      showSegmentationMask(mask, x1, y1, cropWidth, cropHeight);
    };
  };

  const showSegmentationMask = (mask, x1, y1, cropWidth, cropHeight) => {
    const canvas = segCanvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.lineWidth = 1;
    for (let i = 0; i < 16; i++) {
      for (let j = 0; j < 16; j++) {
        const val = mask[i * 16 + j];
        if (val > 0.5) {
          ctx.fillStyle = "rgba(255, 0, 0, 0.4)";
          const px = x1 + (j / 16) * cropWidth;
          const py = y1 + (i / 16) * cropHeight;
          ctx.fillRect(px, py, cropWidth / 16, cropHeight / 16);
        }
      }
    }
  };

  const iou = (a, b) => {
    const interArea = intersection(a, b);
    const unionArea = area(a) + area(b) - interArea;
    return interArea / unionArea;
  };

  const intersection = (a, b) => {
    const [x1, y1, x2, y2] = [
      Math.max(a[0], b[0]),
      Math.max(a[1], b[1]),
      Math.min(a[2], b[2]),
      Math.min(a[3], b[3])
    ];
    return Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  };

  const area = ([x1, y1, x2, y2]) => Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

  return (
    <div style={{ textAlign: "center" }}>
      <h2>YOLOv8 Eye Detection (eye1, eye2)</h2>
      <input type="file" accept="image/*" onChange={handleUpload} />
      <br />
      <canvas ref={canvasRef} style={{ border: "1px solid green", marginTop: 10 }}></canvas>
      <br />
      <canvas ref={segCanvasRef} style={{ border: "1px solid red", marginTop: 10 }}></canvas>
    </div>
  );
};

export default App;
