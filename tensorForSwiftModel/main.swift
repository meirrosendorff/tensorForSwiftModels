//
//  main.swift
//  tensorForSwiftModel
//
//  Created by Meir Rosendorff on 2019/07/15.
//  Copyright © 2019 Meir Rosendorff. All rights reserved.
//

import Foundation
import Python
import TensorFlow

let np = Python.import("numpy")

let rootPath = "/Users/MRosendorff/DVT_Grad_Program/tensorForSwiftModel/tensorForSwiftModel/"

let datSetName = "first2000MFCC"
let testPercentage = 0.2
let hiddenSize: Int = 100
let learningRate: Float = 0.01
let epochCount = 10000
let numBatches = 10
let modelName = "first2000_MFCC_hiddenSize\(hiddenSize)_batches\(numBatches)_epochs\(epochCount)"

//reads in audio files, extracts the features and saves them as .npy files
//Im saving in ./numpyArrays/ NB this directory must exist before you run else it will crash.
_ = parseAudioFilesToNumpyArray(dir: rootPath, indexFile: "audioLog.txt", savedFileName: "numpyArrays/\(datSetName)", maxFiles: 2000)

//create a dataset from the files that where just saved to numpyArrays

print("\nLoading Dataset \(datSetName)")
guard let dataSet = DataSet(datSetName: "numpyArrays/\(datSetName)",
                            testPercentage: testPercentage,
                            numBatches: numBatches) else { fatalError("Unable to build dataset") }

//create a model

print("\nBuilding model for Input Size: \(dataSet.dimOfInput), Output Size: \(dataSet.numLabels), Hidden Size \(hiddenSize)")
var model = BreathingModel(inputSize: dataSet.dimOfInput,
                           outputSize: dataSet.numLabels,
                           hiddenSize: hiddenSize)

print("\nConfiguring optomizer for learning rate: \(learningRate)")
let optimizer = SGD(for: model, learningRate: learningRate)

var trainAccuracyResults: [Float] = []
var trainLossResults: [Float64] = []

var testAccuracyResults: [Float] = []
var testLossResults: [Float64] = []

print("\nBegining Training with \(epochCount) epochs and \(numBatches) batches")

print("\nTraining using \(datSetName) Dataset with \(dataSet.numTrainingFeatures) training examples in \(numBatches) batches and with \(dataSet.numTestingFeatures) testing examples")

for epoch in 1...epochCount {
  var epochLoss: Float64 = 0
  var epochAccuracy: Float = 0

  for batch in dataSet.batches {

    let (loss, grad) = model.valueWithGradient { (model: BreathingModel) -> Tensor<Float64> in
      let logits = model(batch.features)
      return softmaxCrossEntropy(logits: logits, labels: batch.labels)
    }
    optimizer.update(&model.allDifferentiableVariables, along: grad)

    let logits = model(batch.features)
    epochAccuracy += accuracy(predictions: logits.argmax(squeezingAxis: 1), truths: batch.labels) / Float(numBatches)
    epochLoss += loss.scalarized() / Float64(numBatches)

  }

  let testLogits = model(dataSet.ts_features)
  let testLoss = softmaxCrossEntropy(logits: testLogits, labels: dataSet.ts_labels).scalarized()
  let testAccuracy = accuracy(predictions: testLogits.argmax(squeezingAxis: 1), truths: dataSet.ts_labels)

  if epoch > 10 { //ignore the first 10 as they are usually all over the place
    trainAccuracyResults.append(epochAccuracy)
    trainLossResults.append(epochLoss)

    testLossResults.append(testLoss)
    testAccuracyResults.append(testAccuracy)
  }

  if epoch % 50 == 0 {
    print("Epoch \(epoch): Loss: \(epochLoss),\tAccuracy: \(epochAccuracy):\tTest: Loss: \(testLoss),\tAccuracy: \(testAccuracy)")

    print("Training Confusion Matrix")
    printConfusionMatrix(numLabels: dataSet.numLabels, truth: dataSet.tr_labels, prediction: model(dataSet.tr_features).argmax(squeezingAxis: 1))
    print("Testing Confusion Matrix")
    printConfusionMatrix(numLabels: dataSet.numLabels, truth: dataSet.ts_labels, prediction: model(dataSet.ts_features).argmax(squeezingAxis: 1))
  }
}

print("\nSaving Model coreMlModels\(modelName).mlmodel")

model.save(withName: rootPath + "coreMlModels/\(modelName)")


print("\nGathering Data to display and building graphs")
let testLogits = model(dataSet.ts_features)

printConfusionMatrix(numLabels: dataSet.numLabels, truth: dataSet.ts_labels, prediction: testLogits.argmax(squeezingAxis: 1))

buildPlot(tr_Accuracy: trainAccuracyResults, tr_Loss: trainLossResults, ts_Accuracy: testAccuracyResults, ts_Loss: testLossResults)
