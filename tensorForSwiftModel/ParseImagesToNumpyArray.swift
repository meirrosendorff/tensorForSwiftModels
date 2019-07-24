//
//  ParseAudioToNumpyArray.swift
//  tensorForSwiftModel
//
//  Created by Meir Rosendorff on 2019/07/22.
//  Copyright © 2019 Meir Rosendorff. All rights reserved.
//
import Foundation
import TensorFlow
import Python
import AppKit

//use max files if you dont want to read in all files
func parseImagesToNumpyArray(dir: String, savedFileName: String? = nil, maxFilesPerCat maxFiles: Int? = nil) {
  
  let np = Python.import("numpy")
  let os = Python.import("os")
  let cv2 = Python.import("cv2")
  
  let dim = [20, 20]
  
  var features = np.empty(Python.tuple([dim[0], dim[1], 3]))
  var labels = np.empty(0)
  
  var featureContainer = [PythonObject]()
  var labelContainer = [Int]()
  
  var numProccesed = 0
  
  for featureDir in os.listdir(dir) {
    
    let label = getLabelForLetter(letter: String(featureDir)!)

    for imageFile in os.listdir("\(dir)\(featureDir)") where String(imageFile) != nil {
      
      print("Proccesing audio \(numProccesed) for \(featureDir)")

      let img = cv2.imread("\(dir)\(featureDir)/\(String(imageFile)!)")
      let correctlySized = cv2.resize(img, Python.tuple(dim), interpolation: cv2.INTER_AREA)
      featureContainer.append(correctlySized)
      labelContainer.append(label)

      numProccesed += 1

      if let limit = maxFiles, numProccesed > limit { numProccesed = 0; break }
    }
  }
  
  print("Concatinating features into numpy arrays")
  
  features = np.vstack([featureContainer])
  print(features.shape)
  labels = np.array(labelContainer)

  print("Saving")
  if let name = savedFileName {
    np.save("\(name)_features.npy", features)
    print("Saved Features as \(name)_features.npy")
    np.save("\(name)_labels.npy", labels)
    print("Saved labels as \(name)_labels.npy")
  }
}

func getLabelForLetter(letter: String) -> Int {
  
  let alphabet: [String] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".compactMap({ $0.description })
  let features: [String] = alphabet + ["del", "space", "nothing"]
  
  return features.index(of: letter) ?? -1
}
