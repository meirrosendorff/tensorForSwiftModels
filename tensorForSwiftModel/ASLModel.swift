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

struct ASLModel: Layer {
  
  var layer0: Flatten<Float64>
  var layer1: Dense<Float64>
  var layer2: Dense<Float64>
  var layer3: Dense<Float64>

  @noDerivative let inputSize: Int
  @noDerivative let hiddenSize: Int
  @noDerivative let outputSize: Int
  @noDerivative let shape: [Int]
  
  init(inputDim: [Int], outputSize: Int, hiddenSize: Int, savedModel: String? = nil) {
    
    self.inputSize = inputDim.reduce(1, *)
    self.shape = inputDim
    self.hiddenSize = hiddenSize
    self.outputSize = outputSize
    
    
    layer0 = Flatten<Float64>()
    layer1 = Dense<Float64>(inputSize: inputSize, outputSize: hiddenSize, activation: relu)
    layer2 = Dense<Float64>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    layer3 = Dense<Float64>(inputSize: hiddenSize, outputSize: outputSize, activation: softmax)
    
    if let modelName = savedModel {

      print("Loading saved model from \(modelName)\n")
      loadSavedKerasModel(withName: modelName)
    }
    
  }
  
  func save(withName name: String) {
    
    let model = getKerasModel()
    
    model.layers[1].set_weights([layer1.weight.makeNumpyArray(), layer1.bias.makeNumpyArray()])
    model.layers[2].set_weights([layer2.weight.makeNumpyArray(), layer2.bias.makeNumpyArray()])
    model.layers[3].set_weights([layer3.weight.makeNumpyArray(), layer3.bias.makeNumpyArray()])
    
    model.save_weights("\(name).h5")

    let coremltools = Python.import("coremltools")
    
    let alphabet: [String] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".compactMap({ $0.description })
    let labelNames = alphabet + ["del", "space", "nothing"]
    
    let coreMLModel = coremltools.converters.keras.convert(model,
                                                           input_names: "image",
                                                           image_input_names: "image",
                                                           output_names: "output",
                                                           class_labels: labelNames,
                                                           is_bgr: true)
    coreMLModel.author = "mrosendorff"
    coreMLModel.license = "mrosendorff"
    coreMLModel.short_description = "ASL Alphabet Recognition"
    coreMLModel.input_description["image"] = "Image Of Hand"
    coreMLModel.output_description["output"] = "Probability of each letter"
    coreMLModel.output_description["classLabel"] = "Letter"
    
    coreMLModel.save("\(name).mlmodel")
  }
  
  private func getKerasModel() -> PythonObject {
    
    let keras = Python.import("keras")
    
    let model = keras.Sequential()
    
    model.add(keras.layers.Flatten(input_shape: Python.tuple(shape)))
    model.add(keras.layers.Dense(hiddenSize, activation: "relu"))
    model.add(keras.layers.Dense(hiddenSize, activation: "relu"))
    model.add(keras.layers.Dense(outputSize, activation: "softmax"))
    
    return model
  }
  
  private mutating func loadSavedKerasModel(withName name: String) {
    
    let model = getKerasModel()
    
    model.load_weights("\(name).h5")

    setLayerFromKeras(forLayer: &layer1, forKerasLayer: model.layers[1])
    setLayerFromKeras(forLayer: &layer2, forKerasLayer: model.layers[2])
    setLayerFromKeras(forLayer: &layer3, forKerasLayer: model.layers[3])
  
  }
  
  private func setLayerFromKeras( forLayer layer: inout Dense<Float64>, forKerasLayer kerasLayer: PythonObject) {
    
    let np = Python.import("numpy")
    let weightArr = np.array(kerasLayer.get_weights()[0])
    let biasArr = np.array(kerasLayer.get_weights()[1])
    
    guard let weight: Tensor<Float64> = Tensor(numpy: np.float64(weightArr)) else { print("Unable to Load from model");return }
    guard let bias = Tensor<Float64>(numpy: np.float64(biasArr)) else { print("Unable to Load from model");return }
    

    layer.weight = weight
    layer.bias = bias
  }
  
  @differentiable
  func callAsFunction(_ input: Tensor<Float64>) -> Tensor<Float64> {
    return input.sequenced(through: layer0, layer1, layer2, layer3)
  }
}
