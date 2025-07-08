import React, { useRef, useState } from "react";
import * as ort from "onnxruntime-web";

const YOLO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
  'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
  'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
  'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
  'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
  'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

const App = () => {
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);

  const loadModel = async () => {
    if (!model) {
      const session = await ort.InferenceSession.create("yolov8m.onnx");
      setModel(session);
    }
  };

 const handleUpload = async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  // Ensure model is loaded (don't rely on setModel since it's async)
  let session = model;
  if (!session) {
    session = await ort.InferenceSession.create("yolov8m.onnx");
    setModel(session); // store in state
  }

  const [input, imgWidth, imgHeight] = await prepareInput(file);
  const tensor = new ort.Tensor("float32", Float32Array.from(input), [1, 3, 640, 640]);
  const results = await session.run({ images: tensor }); // use loaded session
  const boxes = processOutput(results.output0.data, imgWidth, imgHeight);
  drawImageAndBoxes(file, boxes);
};


  const prepareInput = (file) => {
    return new Promise((resolve) => {
      const img = new Image();
      img.src = URL.createObjectURL(file);
      img.onload = () => {
        const [imgWidth, imgHeight] = [img.width, img.height];
        const canvas = document.createElement("canvas");
        canvas.width = 640;
        canvas.height = 640;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, 640, 640);
        const imageData = ctx.getImageData(0, 0, 640, 640).data;

        const red = [], green = [], blue = [];
        for (let i = 0; i < imageData.length; i += 4) {
          red.push(imageData[i] / 255.0);
          green.push(imageData[i + 1] / 255.0);
          blue.push(imageData[i + 2] / 255.0);
        }
        const input = [...red, ...green, ...blue];
        resolve([input, imgWidth, imgHeight]);
      };
    });
  };

  const runInference = async (input) => {
    const tensor = new ort.Tensor("float32", Float32Array.from(input), [1, 3, 640, 640]);
    const results = await model.run({ images: tensor });
    return results.output0.data;
  };

  const processOutput = (output, imgWidth, imgHeight) => {
    let boxes = [];
    for (let i = 0; i < 8400; i++) {
      const [classId, prob] = [...Array(80).keys()]
        .map((j) => [j, output[8400 * (j + 4) + i]])
        .reduce((a, b) => (b[1] > a[1] ? b : a), [0, 0]);

      if (prob < 0.5) continue;

      const xc = output[i];
      const yc = output[8400 + i];
      const w = output[2 * 8400 + i];
      const h = output[3 * 8400 + i];
      const x1 = (xc - w / 2) / 640 * imgWidth;
      const y1 = (yc - h / 2) / 640 * imgHeight;
      const x2 = (xc + w / 2) / 640 * imgWidth;
      const y2 = (yc + h / 2) / 640 * imgHeight;
      const label = YOLO_CLASSES[classId];
      boxes.push([x1, y1, x2, y2, label, prob]);
    }

    boxes = boxes.sort((a, b) => b[5] - a[5]);
    const result = [];
    while (boxes.length > 0) {
      result.push(boxes[0]);
      boxes = boxes.filter((box) => iou(boxes[0], box) < 0.7);
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

      boxes.forEach(([x1, y1, x2, y2, label]) => {
        ctx.strokeStyle = "#00FF00";
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = "#00FF00";
        ctx.fillRect(x1, y1 - 20, ctx.measureText(label).width + 10, 20);
        ctx.fillStyle = "#000000";
        ctx.fillText(label, x1 + 5, y1 - 5);
      });
    };
  };

  const iou = (box1, box2) => {
    const intersect = intersection(box1, box2);
    return intersect / (area(box1) + area(box2) - intersect);
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
      <h2>YOLOv8 Object Detection (ONNX)</h2>
      <input type="file" accept="image/*" onChange={handleUpload} />
      <br />
      <canvas ref={canvasRef} style={{ border: "1px solid black", marginTop: 10 }}></canvas>
    </div>
  );
};

export default App;
