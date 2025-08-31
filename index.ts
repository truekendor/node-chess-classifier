import * as tf from "@tensorflow/tfjs";

//
//
import { readdir } from "node:fs/promises";
import path from "node:path";

//
//
import { createCanvas, Image, loadImage } from "canvas";
import {
  calculateImageFeatures,
  loadMobileNetModel,
  setupModel as setupTransferHeadModel,
} from "./NNUtils.js";
import { train as trainModel } from "./train.js";
import { chessPiecesLookup } from "./piecesLookup.js";

const config = {
  datasetPath: "./dataset/tiles",
} as const;

// models
const mobilenet = await loadMobileNetModel();
export const model = setupTransferHeadModel();

const trainingDataInputs: tf.Tensor1D[] = [];
// int 0..=12
const trainingDataOutputs: number[] = [];

main();
async function main() {
  await extractImageFeaturesFromDataset();
  await trainModel({
    model,
    trainingDataInputs,
    trainingDataOutputs,
  });
}

async function extractImageFeaturesFromDataset() {
  const datasetPath = config.datasetPath;
  const tileImages = await readdir(config.datasetPath);

  const canvas = createCanvas(2000, 2000);
  const c = canvas.getContext("2d");

  const promiseList: Promise<Image>[] = [];

  for (let i = 0; i < tileImages.length; i++) {
    const currentTileImage = tileImages[i];

    const tileImage = loadImage(path.join(datasetPath, currentTileImage));
    promiseList.push(tileImage);
  }

  const result = await Promise.all(promiseList);

  for (let i = 0; i < result.length; i++) {
    const tileImage = result[i];
    const currentTileImage = tileImages[i];

    const { naturalWidth, naturalHeight } = tileImage;

    console.log("extracting features...", i);

    canvas.width = naturalWidth;
    canvas.height = naturalHeight;

    c.drawImage(tileImage, 0, 0);

    const pieceType = currentTileImage.split(
      "_"
    )[1][0] as keyof typeof chessPiecesLookup;
    const tileFeature = calculateImageFeatures({
      canvas,
      mobilenet,
    })!;

    trainingDataInputs.push(tileFeature);

    const pieceId = chessPiecesLookup[pieceType];

    console.log("piece id: ", pieceId, pieceType);
    if (pieceId) {
      trainingDataOutputs.push(pieceId);
    } else {
      console.log("Unexpected piece type");
    }
  }
}
