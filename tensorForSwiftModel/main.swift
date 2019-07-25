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

let savedModel = "\(rootPath)coreMLModels/first_200_Gray_Scaled_fingers_2828_batches100_epochs200_test59_train72"

let dataSetName = "all_Gray_Scaled_fingers_2828"
let testPercentage = 0.2
let learningRate: Float = 0.1
let epochCount = 200
let numBatches = 100
let modelName = "\(dataSetName)_batches\(numBatches)_epochs\(epochCount)"

//reads in audio files, extracts the features and saves them as .npy files
//Im saving in ./numpyArrays/ NB this directory must exist before you run else it will crash.
//parseImagesToNumpyArray(dir: rootPath + "fingerSpelling/", savedFileName: "\(rootPath)numpyArrays/\(dataSetName)", maxFilesPerCat: 200)

//create a dataset from the files that where just saved to numpyArrays

print("\nLoading Dataset \(dataSetName)")
guard let dataSet = DataSet(datSetName: "numpyArrays/\(dataSetName)",
                            testPercentage: testPercentage,
                            numBatches: numBatches) else { fatalError("Unable to build dataset") }

//create a model

print("\nBuilding model for Input Size: \(dataSet.dimOfInput), Output Size: \(dataSet.numLabels)")
var model = ASLModel(inputDim: dataSet.dimOfInput,
                           outputSize: dataSet.numLabels,
                           savedModel: savedModel)

print("\nConfiguring optomizer for learning rate: \(learningRate)")
let optimizer = SGD(for: model, learningRate: learningRate)

var trainAccuracyResults: [Float] = []
var trainLossResults: [Float64] = []

var testAccuracyResults: [Float] = []
var testLossResults: [Float64] = []

print("\nBegining Training with \(epochCount) epochs and \(numBatches) batches")

print("\nTraining using \(dataSetName) Dataset with \(dataSet.numTrainingFeatures) training examples in \(numBatches) batches and with \(dataSet.numTestingFeatures) testing examples\n")

for epoch in 1...epochCount {
  var epochLoss: Float64 = 0
  var epochAccuracy: Float = 0

  for batch in dataSet.batches {

    let (loss, grad) = model.valueWithGradient { (model: ASLModel) -> Tensor<Float64> in
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

  print("Epoch \(epoch): Loss: \(epochLoss),\tAccuracy: \(epochAccuracy):\tTest: Loss: \(testLoss),\tAccuracy: \(testAccuracy)")
  if epoch % 50 == 0 {

    print("Testing Confusion Matrix")
    printConfusionMatrix(numLabels: dataSet.numLabels, truth: dataSet.ts_labels, prediction: model(dataSet.ts_features).argmax(squeezingAxis: 1))
    model.save(withName: rootPath + "coreMlModels/trainingLogs/\(modelName)_epoch\(epoch)_test\(Int(testAccuracyResults[testAccuracyResults.count - 1]*100))_train\(Int(trainAccuracyResults[trainAccuracyResults.count - 1]*100))")
  }
}

print("\nSaving Model coreMlModels\(modelName).mlmodel")

model.save(withName: rootPath + "coreMlModels/\(modelName)_test\(Int(testAccuracyResults[testAccuracyResults.count - 1]*100))_train\(Int(trainAccuracyResults[trainAccuracyResults.count - 1]*100))")


print("\nGathering Data to display and building graphs")
let testLogits = model(dataSet.ts_features)

printConfusionMatrix(numLabels: dataSet.numLabels, truth: dataSet.ts_labels, prediction: testLogits.argmax(squeezingAxis: 1))

buildPlot(tr_Accuracy: trainAccuracyResults, tr_Loss: trainLossResults, ts_Accuracy: testAccuracyResults, ts_Loss: testLossResults)
