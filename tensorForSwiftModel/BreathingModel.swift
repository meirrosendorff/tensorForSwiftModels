//
//  breathingModel.swift
//  tensorForSwiftModel
//
//  Created by Meir Rosendorff on 2019/07/16.
//  Copyright © 2019 Meir Rosendorff. All rights reserved.
//

import Foundation
import TensorFlow


struct BreathingModel: Layer {
  
  var layer1: Dense<Float64>
  var layer2: Dense<Float64>
  var layer3: Dense<Float64>

  
  init(inputSize: Int, outputSize: Int, hiddenSize: Int) {
    
    layer1 = Dense<Float64>(inputSize: inputSize, outputSize: hiddenSize, activation: relu)
    layer2 = Dense<Float64>(inputSize: hiddenSize, outputSize: hiddenSize, activation: softmax)
    layer3 = Dense<Float64>(inputSize: hiddenSize, outputSize: outputSize)
  }
  
  @differentiable
  func callAsFunction(_ input: Tensor<Float64>) -> Tensor<Float64> {
    return input.sequenced(through: layer1, layer2, layer3)
  }
}
