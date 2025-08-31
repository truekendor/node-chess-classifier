import * as tf from "@tensorflow/tfjs";

import { chessPiecesLookup } from "./piecesLookup.js";
import { Canvas } from "canvas";

export const MOBILE_NET_INPUT_WIDTH = 224;

export function setupModel() {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [1024],
      units: 512,
      activation: "relu",
    })
  );

  model.add(
    tf.layers.dense({
      inputShape: [512],
      units: 64,
      activation: "relu",
    })
  );

  // output layer
  model.add(
    tf.layers.dense({
      units: Object.keys(chessPiecesLookup).length,
      activation: "softmax",
    })
  );

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

export async function loadMobileNetModel() {
  console.log("🟡 Loading Mobilenet...");

  const MODEL_URL = `https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1`;

  const mobilenet = await tf.loadGraphModel(MODEL_URL, {
    fromTFHub: true,
  });

  tf.tidy(() => {
    // warmup mobilenet model
    // deleting this line will cause lag
    // on first prediction
    mobilenet.predict(
      tf.zeros([1, MOBILE_NET_INPUT_WIDTH, MOBILE_NET_INPUT_WIDTH, 3])
    );
  });

  console.log("✅ Mobilenet ready");

  return mobilenet;
}

export function calculateImageFeatures({
  canvas,
  mobilenet,
}: {
  mobilenet: tf.GraphModel;
  canvas: Canvas;
}) {
  try {
    return tf.tidy(() => {
      // @ts-ignore
      const canvasAsTensor = tf.browser.fromPixels(canvas);

      // Resize image to mobilenet size
      const resizedTensorFrame = tf.image.resizeBilinear(
        canvasAsTensor,
        [MOBILE_NET_INPUT_WIDTH, MOBILE_NET_INPUT_WIDTH],
        true
      );

      // tensors normalization [0, 1]
      const normalizedTensorFrame = resizedTensorFrame.div(255);

      const result = mobilenet
        .predict(normalizedTensorFrame.expandDims())
        // @ts-ignore
        .squeeze();

      return result as tf.Tensor1D;
    });
  } catch (e) {
    // @ts-ignore
    console.log(e?.message);
  }
}
