import * as tf from "@tensorflow/tfjs";
import { chessPiecesLookup } from "./piecesLookup.js";
import { model } from "./index.js";

type TrainInput = {
  trainingDataInputs: Array<tf.Tensor1D>;
  trainingDataOutputs: Array<number>;
  model: tf.Sequential;
};

export async function train({
  trainingDataInputs,
  trainingDataOutputs,
  model,
}: TrainInput) {
  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);

  const outputsAsTensor = tf.tensor1d(trainingDataOutputs, "int32");
  const oneHotOutputs = tf.oneHot(
    outputsAsTensor,
    Object.keys(chessPiecesLookup).length
  );
  const inputsAsTensor = tf.stack(trainingDataInputs);

  await model.fit(inputsAsTensor, oneHotOutputs, {
    shuffle: true,
    batchSize: 8,
    epochs: 5,
    validationSplit: 0.3,

    callbacks: {
      onEpochEnd: logProgress,
    },
  });

  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();
}

function logProgress(epoch: number, logs: any): void {
  let highestValAcc = 0.0;

  console.log(`epoch number: ${epoch}`);
  console.table(logs);

  if (highestValAcc) {
    highestValAcc = logs.val_acc;

    model.save(`file://./model_highest_acc`);
  }
}
