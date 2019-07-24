//
//  outputGraphicsHelpers.swift
//  tensorForSwiftModel
//
//  Created by Meir Rosendorff on 2019/07/17.
//  Copyright © 2019 Meir Rosendorff. All rights reserved.
//

import Foundation
import Python
import TensorFlow

func accuracy(predictions: Tensor<Int32>, truths: Tensor<Int32>) -> Float {
  return Tensor<Float>(predictions .== truths).mean().scalarized()
}

func buildPlot(tr_Accuracy: [Float], tr_Loss: [Float64], ts_Accuracy: [Float], ts_Loss: [Float64]) {
  
  let plt = Python.import("matplotlib.pyplot")
  
  plt.figure(figsize: [12, 8])
  
  let accuracyAxes = plt.subplot(2, 2, 1)
  accuracyAxes.set_ylabel("Training Accuracy")
  accuracyAxes.plot(tr_Accuracy)
  
  let lossAxes = plt.subplot(2, 2, 2)
  lossAxes.set_ylabel("Training Loss")
  lossAxes.plot(tr_Loss)
  
  let testAccuracy = plt.subplot(2, 2, 3)
  testAccuracy.set_ylabel("Test Accuracy")
  testAccuracy.plot(ts_Accuracy)
  
  let testLoss = plt.subplot(2, 2, 4)
  testLoss.set_ylabel("Test Loss")
  testLoss.plot(ts_Loss)
  testLoss.set_xlabel("Epoch")
  
  plt.show()
}

func printConfusionMatrix(numLabels: Int, truth: Tensor<Int32>, prediction: Tensor<Int32>) {
  
  let truthArr = truth.makeNumpyArray()
  let predictionArr = prediction.makeNumpyArray()
  
  var matrix = Array(repeating: Array(repeating: 0, count: numLabels), count: numLabels)
  
  for (truth, prediction) in zip(truthArr, predictionArr) {
    
    guard let y = Int(truth), let x = Int(prediction) else { continue }
    
    if x >= numLabels || y >= numLabels {
      print("Invalid number of labels compared to given labels")
      return
    }
    
    matrix[y][x] += 1
  }
  
  print("\n***********************************************************")
  print("Accuracy: \(accuracy(predictions: prediction, truths: truth))\n")

  var toPrint = "\t\t\t"

  for i in 0..<numLabels { toPrint +=  i < 10 ? "\(i)\t\t\t" : "\(i)\t\t"}
  toPrint += "\n\t\t\t"
  for _ in 0..<numLabels { toPrint += "-\t\t\t"}

  for (i, row) in matrix.enumerated() {

    toPrint += "\n\(i):\t\t"
    
    for column in row {
      
      if column >= 1000 {
        toPrint += "\(column)\t"
      } else if column >= 10 {
        toPrint += "\(column)\t\t"
      } else {
        toPrint += "\(column)\t\t\t"
      }
    }
  }

  print(toPrint)

  print("\n***********************************************************")
}
