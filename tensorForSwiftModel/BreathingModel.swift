//
//  breathingModel.swift
//  tensorForSwiftModel
//
//  Created by Meir Rosendorff on 2019/07/16.
//  Copyright © 2019 Meir Rosendorff. All rights reserved.
//

import Foundation
import TensorFlow
import Python

struct BreathingModel: Layer {
  
  var layer1: Dense<Float64>
  var layer2: Dense<Float64>
  var layer3: Dense<Float64>

  @noDerivative let inputSize: Int
  @noDerivative let hiddenSize: Int
  @noDerivative let outputSize: Int
  
  init(inputSize: Int, outputSize: Int, hiddenSize: Int) {
    
    self.inputSize = inputSize
    self.hiddenSize = hiddenSize
    self.outputSize = outputSize
    
    layer1 = Dense<Float64>(inputSize: inputSize, outputSize: hiddenSize, activation: relu)
    layer2 = Dense<Float64>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    layer3 = Dense<Float64>(inputSize: hiddenSize, outputSize: outputSize, activation: softmax)
    
  }
  
  func save(withName name: String) {
    
    let keras = Python.import("keras")
    
    let model = keras.Sequential()
    
    model.add(keras.layers.Dense(hiddenSize, activation: "relu", input_shape: [inputSize,]))
    model.add(keras.layers.Dense(hiddenSize, activation: "relu"))
    model.add(keras.layers.Dense(outputSize, activation: "softmax"))
    
    model.layers[0].set_weights([layer1.weight.makeNumpyArray(), layer1.bias.makeNumpyArray()])
    model.layers[1].set_weights([layer2.weight.makeNumpyArray(), layer2.bias.makeNumpyArray()])
    model.layers[2].set_weights([layer3.weight.makeNumpyArray(), layer3.bias.makeNumpyArray()])

    let coremltools = Python.import("coremltools")
    
    let coreMLModel = coremltools.converters.keras.convert(model)
    
    coreMLModel.save("\(name).mlmodel")
  }
  
  @differentiable
  func callAsFunction(_ input: Tensor<Float64>) -> Tensor<Float64> {
    return input.sequenced(through: layer1, layer2, layer3)
  }
}
