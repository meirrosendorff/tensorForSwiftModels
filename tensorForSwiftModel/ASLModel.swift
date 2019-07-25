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
  
  var layer0: Conv2D<Float64>
  var layer1: MaxPool2D<Float64>
  var layer2: Dropout<Float64>
  var layer3: Conv2D<Float64>
  var layer4: MaxPool2D<Float64>
  var layer5: Dropout<Float64>
  var layer6: Conv2D<Float64>
  var layer7: MaxPool2D<Float64>
  var layer8: Dropout<Float64>
  var layer9: Flatten<Float64>
  var layer10: Dense<Float64>
  var layer11: Dense<Float64>

  @noDerivative let inputSize: Int
  @noDerivative let outputSize: Int
  @noDerivative let shape: [Int]
  
  init(inputDim: [Int], outputSize: Int, savedModel: String? = nil) {
    
    self.inputSize = inputDim.reduce(1, *)
    self.shape = inputDim
    self.outputSize = outputSize

    layer0 = Conv2D<Float64>(filterShape: (5, 5, 1, 32), strides: (1, 1), padding: .valid, activation: relu)
    layer1 = MaxPool2D<Float64>(poolSize: (2, 2), strides: (2, 2), padding: .valid)
    layer2 = Dropout<Float64>(probability: 0.5)
    layer3 = Conv2D<Float64>(filterShape: (3, 3, 32, 64), strides: (1, 1), padding: .valid, activation: relu)
    layer4 = MaxPool2D<Float64>(poolSize: (2, 2), strides: (2, 2), padding: .valid)
    layer5 = Dropout<Float64>(probability: 0.2)
    layer6 = Conv2D<Float64>(filterShape: (1, 1, 64, 128), strides: (1, 1), padding: .valid, activation: relu)
    layer7 = MaxPool2D<Float64>(poolSize: (2, 2), strides: (2, 2), padding: .valid)
    layer8 = Dropout<Float64>(probability: 0.2)
    layer9 = Flatten<Float64>()
    layer10 = Dense<Float64>(inputSize: 512, outputSize: 128, activation: relu)
    layer11 = Dense<Float64>(inputSize: 128, outputSize: outputSize, activation: softmax)

    if let modelName = savedModel {

      print("Loading saved model from \(modelName)\n")
      loadSavedKerasModel(withName: modelName)
    }
    
  }
  
  func save(withName name: String) {
    
    let model = getKerasModel()
    
    model.layers[0].set_weights([layer0.filter.makeNumpyArray(), layer0.bias.makeNumpyArray()])
    model.layers[3].set_weights([layer3.filter.makeNumpyArray(), layer3.bias.makeNumpyArray()])
    model.layers[6].set_weights([layer6.filter.makeNumpyArray(), layer6.bias.makeNumpyArray()])
    model.layers[10].set_weights([layer10.weight.makeNumpyArray(), layer10.bias.makeNumpyArray()])
    model.layers[11].set_weights([layer11.weight.makeNumpyArray(), layer11.bias.makeNumpyArray()])
    
    model.save_weights("\(name).h5")

    let coremltools = Python.import("coremltools")

    let labelNames: [String] = "abcdefghiklmnopqrstuvwxy".compactMap({ $0.description })

    let coreMLModel = coremltools.converters.keras.convert(model,
                                                           input_names: "image",
                                                           image_input_names: "image",
                                                           output_names: "output",
                                                           class_labels: labelNames,
                                                           is_bgr: true,
                                                           image_scale: 1/255.0)
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

    model.add(keras.layers.Conv2D(32, Python.tuple([5, 5]), strides: 1, activation: "relu", padding: "valid", input_shape: self.shape))
    model.add(keras.layers.MaxPooling2D(pool_size: Python.tuple([2, 2]), strides: 2, padding: "valid"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Conv2D(64, Python.tuple([3, 3]), strides: 1, activation: "relu", padding: "valid"))
    model.add(keras.layers.MaxPooling2D(pool_size: Python.tuple([2, 2]), strides: 2, padding: "valid"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv2D(128, Python.tuple([1, 1]), strides: 1, activation: "relu", padding: "valid"))
    model.add(keras.layers.MaxPooling2D(pool_size: Python.tuple([2, 2]), strides: 2, padding: "valid"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation: "relu"))
    model.add(keras.layers.Dense(outputSize, activation: "softmax"))

    return model
  }
  
  private mutating func loadSavedKerasModel(withName name: String) {
    
    let model = getKerasModel()

    model.load_weights("\(name).h5")

    setLayerFromKeras(forLayer: &layer0, forKerasLayer: model.layers[0])
    setLayerFromKeras(forLayer: &layer3, forKerasLayer: model.layers[3])
    setLayerFromKeras(forLayer: &layer6, forKerasLayer: model.layers[6])
    setLayerFromKeras(forLayer: &layer10, forKerasLayer: model.layers[10])
    setLayerFromKeras(forLayer: &layer11, forKerasLayer: model.layers[11])

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
  
  private func setLayerFromKeras( forLayer layer: inout Conv2D<Float64>, forKerasLayer kerasLayer: PythonObject) {
    
    let np = Python.import("numpy")
    let weightArr = np.array(kerasLayer.get_weights()[0])
    let biasArr = np.array(kerasLayer.get_weights()[1])
    
    guard let weight: Tensor<Float64> = Tensor(numpy: np.float64(weightArr)) else { print("Unable to Load from model");return }
    guard let bias = Tensor<Float64>(numpy: np.float64(biasArr)) else { print("Unable to Load from model");return }
    
    
    layer.filter = weight
    layer.bias = bias
  }
  
  @differentiable
  func callAsFunction(_ input: Tensor<Float64>) -> Tensor<Float64> {

    let layer0Result = layer0(input)
//    print(layer0Result.shape)
    let layer1Result = layer1(layer0Result)
//    print(layer1Result.shape)
    let layer2Result = layer2(layer1Result)
//    print(layer2Result.shape)
    let layer3Result = layer3(layer2Result)
//    print(layer3Result.shape)
    let layer4Result = layer4(layer3Result)
//    print(layer4Result.shape)
    let layer5Result = layer5(layer4Result)
//    print(layer5Result.shape)
    let layer6Result = layer6(layer5Result)
//    print(layer6Result.shape)
    let layer7Result = layer7(layer6Result)
//    print(layer7Result.shape)
    let layer8Result = layer8(layer7Result)
//    print(layer8Result.shape)
    let layer9Result = layer9(layer8Result)
//    print(layer9Result.shape)
    let layer10Result = layer10(layer9Result)
//    print(layer10Result.shape)
    let layer11Result = layer11(layer10Result)
//    print(layer11Result.shape)
    return layer11Result
  }
}
