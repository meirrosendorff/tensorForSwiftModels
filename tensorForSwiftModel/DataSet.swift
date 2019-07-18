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
  
  let labelNames = ["Healthy", "Crackles", "Wheezing", "Crackles + Wheezing"]
  let tr_features, ts_features: Tensor<Float64>
  let tr_labels, ts_labels: Tensor<Int32>
  var batches: [(features: Tensor<Float64>, labels: Tensor<Int32>)]
  var numTrainingFeatures: Int
  var numTestingFeatures: Int
  var dimOfInput: Int
  var numLabels: Int { return labelNames.count}

  init?(datSetName: String, testPercentage: Double, numBatches: Int) {
    
    let np = Python.import("numpy")
    let skLearn = Python.import("sklearn.model_selection")
    
    let labels = np.load(rootPath + "\(datSetName)_labels.npy")
    let features = np.load(rootPath + "\(datSetName)_features.npy")
    
    guard let dim = Int(features.shape.tuple2.1) else { print("Unable to get feature Dim"); return nil }
    dimOfInput = dim
    
    let (tr_features, ts_features, tr_labels, ts_labels) =
      skLearn.train_test_split(features,
                               labels,
                               test_size: testPercentage,
                               random_state: 42).tuple4
    
    self.numTrainingFeatures = tr_features.count
    self.numTestingFeatures = ts_features.count
    
    guard let tr_featuresAttempt = Tensor<Float64>(numpy: tr_features) else { return nil }
    guard let ts_featuresAttempt = Tensor<Float64>(numpy: ts_features) else { return nil }
    guard let tr_labelsAttempt = Tensor<Int32>(numpy: np.int32(tr_labels)) else { return nil }
    guard let ts_labelsAttempt = Tensor<Int32>(numpy: np.int32(ts_labels)) else { return nil }

    let featuresSplit = np.array_split(tr_features, numBatches)
    let labelsSplit = np.array_split(tr_labels, numBatches)
    
    batches = []
    for (feature, label) in zip(featuresSplit, labelsSplit) {
      
      guard let featureAttempt = Tensor<Float64>(numpy: feature) else { return nil }
      guard let labelAttempt = Tensor<Int32>(numpy: np.int32(label)) else { return nil }
      
      batches.append((featureAttempt, labelAttempt))
    }
    
    self.tr_features = tr_featuresAttempt
    self.ts_features = ts_featuresAttempt
    self.tr_labels = tr_labelsAttempt
    self.ts_labels = ts_labelsAttempt
  }
}
