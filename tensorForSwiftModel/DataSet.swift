//
//  DataSet.swift
//  tensorForSwiftModel
//
//  Created by Meir Rosendorff on 2019/07/16.
//  Copyright © 2019 Meir Rosendorff. All rights reserved.
//

import Foundation
import Python
import TensorFlow

struct DataSet {
  
  let labelNames: [String]
  let ts_features: Tensor<Float64>
  let ts_labels: Tensor<Int32>
  var batches: [(features: Tensor<Float64>, labels: Tensor<Int32>)]
  var numTrainingFeatures: Int
  var numTestingFeatures: Int
  var dimOfInput: [Int]
  var numLabels: Int { return labelNames.count}

  init?(datSetName: String, testPercentage: Double, numBatches: Int) {
    
    labelNames = "abcdefghiklmnopqrstuvwxy".compactMap({ $0.description }) + ["nothing"]
    
    let np = Python.import("numpy")
    let skLearn = Python.import("sklearn.model_selection")
    
    print("Loading")
    let labels = np.load(rootPath + "\(datSetName)_labels.npy")
    let features = np.load(rootPath + "\(datSetName)_features.npy")

    let dims = features.shape.tuple4
    dimOfInput = [Int(dims.1)!, Int(dims.2)!, Int(dims.3)!]
    
    print("Spliting")
    
    var tr_features, ts_features, tr_labels, ts_labels: PythonObject!

    (tr_features, ts_features, tr_labels, ts_labels) =
      skLearn.train_test_split(features,
                               labels,
                               test_size: testPercentage,
                               random_state: 42).tuple4
    
    self.numTrainingFeatures = tr_features.count
    self.numTestingFeatures = ts_features.count
    
    print("Building Tensors")
    guard let ts_featuresAttempt = Tensor<Float64>(numpy: np.float64(ts_features)) else { return nil }
    ts_features = nil
    guard let ts_labelsAttempt = Tensor<Int32>(numpy: np.int32(ts_labels)) else { return nil }
    ts_labels = nil

    print("Building Batches")
    let featuresSplit = np.array_split(tr_features, numBatches)
    tr_features = nil
    let labelsSplit = np.array_split(tr_labels, numBatches)
    ts_features = nil
    
    batches = []
    for (feature, label) in zip(featuresSplit, labelsSplit) {
      
      guard let featureAttempt = Tensor<Float64>(numpy: np.float64(feature)) else { return nil }
      guard let labelAttempt = Tensor<Int32>(numpy: np.int32(label)) else { return nil }
      
      batches.append((featureAttempt, labelAttempt))
    }
    
    self.ts_features = ts_featuresAttempt
    self.ts_labels = ts_labelsAttempt
  }
}
