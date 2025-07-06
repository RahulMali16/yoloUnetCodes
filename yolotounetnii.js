import React, { useRef, useState } from 'react';
import * as nifti from 'nifti-reader-js';
import * as ort from 'onnxruntime-web';

const App = () => {
  const canvasRef = useRef(null);
  const [yoloModel, setYoloModel] = useState(null);
  const [unetModel, setUnetModel] = useState(null);

  const handleFile = async (e) => {
    const file = e.target.files[0];
    const buffer = await file.arrayBuffer();

    const header = nifti.readHeader(buffer);
    const raw = new Float32Array(nifti.readImage(header, buffer));
    const W = header.dims[1], H = header.dims[2], D = header.dims[3];

    const yolo = yoloModel || await ort.InferenceSession.create('/yolov8.onnx');
    const unet = unetModel || await ort.InferenceSession.create('/unet.onnx');
    if (!yoloModel) setYoloModel(yolo);
    if (!unetModel) setUnetModel(unet);

    const getSlice = (z) => {
      const slice = new Float32Array(W * H);
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++)
          slice[y * W + x] = raw[z * W * H + y * W + x] || 0;
      const max = Math.max(...slice);
      for (let i = 0; i < slice.length; i++) slice[i] /= max || 1;
      return slice;
    };

    for (let z = 1; z < D - 1; z++) {
      const R = getSlice(z - 1), G = getSlice(z), B = getSlice(z + 1);
      drawSliceToCanvas(G, W, H);  // draw background

      const input = await buildRGBTensor(R, G, B, W, H);
      const tensor = new ort.Tensor('float32', input, [1, 3, 512, 512]);
      const output = await yolo.run({ [yolo.inputNames[0]]: tensor });
      const result = output[yolo.outputNames[0]].data;

      const boxes = parseYOLOOutput(result, W, H);
      if (boxes.length) {
        const { x1, y1, x2, y2, score } = boxes[0];
        console.log(`✅ DETECTED at slice ${z}: x=${x1}, y=${y1}, w=${x2 - x1}, h=${y2 - y1}, conf=${score.toFixed(2)}`);
        drawBoxes(canvasRef.current, [boxes[0]]);

        const cropped = cropSlice(G, W, H, x1, y1, x2, y2);
        const resizedInput = await resizeTo16x16(cropped);
        const unetTensor = new ort.Tensor('float32', resizedInput, [1, 1, 16, 16]);
        const segOut = await unet.run({ [unet.inputNames[0]]: unetTensor });
        const segMask = segOut[unet.outputNames[0]].data;

        overlaySegmentation(canvasRef.current, G, segMask, x1, y1, x2, y2, W, H);
        break;
      }
    }
  };

  const buildRGBTensor = (R, G, B, W, H) => new Promise(resolve => {
    const tmp = document.createElement('canvas');
    tmp.width = W; tmp.height = H;
    const ctx = tmp.getContext('2d');
    const img = ctx.createImageData(W, H);
    for (let i = 0; i < R.length; i++) {
      img.data[i * 4 + 0] = R[i] * 255;
      img.data[i * 4 + 1] = G[i] * 255;
      img.data[i * 4 + 2] = B[i] * 255;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    const resized = document.createElement('canvas');
    resized.width = 512; resized.height = 512;
    resized.getContext('2d').drawImage(tmp, 0, 0, 512, 512);
    const rdata = resized.getContext('2d').getImageData(0, 0, 512, 512).data;

    const floatData = new Float32Array(3 * 512 * 512);
    for (let i = 0; i < 512 * 512; i++) {
      floatData[i] = rdata[i * 4 + 0] / 255;
      floatData[i + 512 * 512] = rdata[i * 4 + 1] / 255;
      floatData[i + 2 * 512 * 512] = rdata[i * 4 + 2] / 255;
    }
    resolve(floatData);
  });

  const parseYOLOOutput = (data, origW, origH) => {
    const boxes = [], scaleX = origW / 512, scaleY = origH / 512;
    for (let i = 0; i < data.length; i += 85) {
      if (data[i + 4] < 0.5) continue;
      const [xc, yc, w, h] = data.slice(i, i + 4);
      const x1 = (xc - w / 2) * scaleX;
      const y1 = (yc - h / 2) * scaleY;
      const x2 = (xc + w / 2) * scaleX;
      const y2 = (yc + h / 2) * scaleY;
      boxes.push({ x1, y1, x2, y2, score: data[i + 4] });
    }
    return boxes;
  };

  const cropSlice = (slice, width, height, x1, y1, x2, y2) => {
    const cropW = x2 - x1, cropH = y2 - y1;
    const cropped = new Float32Array(cropW * cropH);
    for (let y = 0; y < cropH; y++)
      for (let x = 0; x < cropW; x++)
        cropped[y * cropW + x] = slice[(y1 + y) * width + (x1 + x)];
    return { data: cropped, width: cropW, height: cropH };
  };

  const resizeTo16x16 = (cropped) => new Promise(resolve => {
    const tmp = document.createElement('canvas');
    tmp.width = cropped.width;
    tmp.height = cropped.height;
    const ctx = tmp.getContext('2d');
    const img = ctx.createImageData(cropped.width, cropped.height);
    for (let i = 0; i < cropped.data.length; i++) {
      const v = Math.floor(cropped.data[i] * 255);
      img.data.set([v, v, v, 255], i * 4);
    }
    ctx.putImageData(img, 0, 0);

    const resized = document.createElement('canvas');
    resized.width = 16;
    resized.height = 16;
    const rctx = resized.getContext('2d');
    rctx.drawImage(tmp, 0, 0, 16, 16);
    const rdata = rctx.getImageData(0, 0, 16, 16).data;

    const floatData = new Float32Array(16 * 16);
    for (let i = 0; i < 16 * 16; i++) floatData[i] = rdata[i * 4] / 255;
    resolve(floatData);
  });

  const drawSliceToCanvas = (slice, W, H) => {
    const canvas = canvasRef.current;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(W, H);
    for (let i = 0; i < slice.length; i++) {
      const val = Math.floor(slice[i] * 255);
      imgData.data.set([val, val, val, 255], i * 4);
    }
    ctx.putImageData(imgData, 0, 0);
  };

  const drawBoxes = (canvas, boxes) => {
    const ctx = canvas.getContext('2d');
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    boxes.forEach(({ x1, y1, x2, y2 }) => {
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    });
  };

  const overlaySegmentation = (canvas, slice, mask, x1, y1, x2, y2, W, H) => {
    const ctx = canvas.getContext('2d');
    const img = ctx.getImageData(0, 0, W, H);
    const cropW = x2 - x1;
    const cropH = y2 - y1;
    for (let y = 0; y < cropH; y++) {
      for (let x = 0; x < cropW; x++) {
        const mx = Math.floor(x * 16 / cropW);
        const my = Math.floor(y * 16 / cropH);
        const maskVal = mask[my * 16 + mx];
        if (maskVal > 0.5) {
          const idx = ((y1 + y) * W + (x1 + x)) * 4;
          img.data[idx + 0] = 255; // red
          img.data[idx + 1] = 0;
          img.data[idx + 2] = 0;
          img.data[idx + 3] = 180;
        }
      }
    }
    ctx.putImageData(img, 0, 0);
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Organ Detection + Segmentation (.nii)</h2>
      <input type="file" accept=".nii,.nii.gz" onChange={handleFile} />
      <canvas ref={canvasRef} style={{ marginTop: 10, border: '1px solid black' }} />
    </div>
  );
};

export default App;
