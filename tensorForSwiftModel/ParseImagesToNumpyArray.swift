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
  let random = Python.import("random")
  
  let dim = [28, 28]
  
  var features = np.empty(Python.tuple([dim[0], dim[1]]))
  var labels = np.empty(0)
  
  var featureContainer = [PythonObject]()
  var labelContainer = [Int]()
  
  var numProccesed = 0
  
  for featureDir in os.listdir(dir) {
    
    let label = getLabelForLetter(letter: String(featureDir)!)
    
    let imageFiles = os.listdir("\(dir)\(featureDir)") //ensure good mix of all examples when taking less than everything
    random.Random(42).shuffle(imageFiles)
    for imageFile in  imageFiles where String(imageFile) != nil {
      
      print("Proccesing audio \(numProccesed) for \(featureDir)")

      let img = cv2.imread("\(dir)\(featureDir)/\(String(imageFile)!)")
      let correctlySized = cv2.resize(img, Python.tuple(dim), interpolation: cv2.INTER_AREA)
      let gray = cv2.cvtColor(correctlySized, cv2.COLOR_BGR2GRAY)
//      let correctShape = np.transpose(gray, Python.tuple([2, 1, 0]))
      let scaledDown = gray / 255
      featureContainer.append(scaledDown)
      labelContainer.append(label)

      numProccesed += 1

      if let limit = maxFiles, numProccesed > limit { numProccesed = 0; break }
    }
  }
  
  print("Concatinating features into numpy arrays")
  
  features = np.vstack([featureContainer])
  print(features.shape)
  let reshapedFeatures = features.reshape(features.shape[0], dim[0], dim[1], 1)
  print(reshapedFeatures.shape)
  labels = np.array(labelContainer)

  print("Saving")
  if let name = savedFileName {
    np.save("\(name)_features.npy", reshapedFeatures)
    print("Saved Features as \(name)_features.npy")
    np.save("\(name)_labels.npy", labels)
    print("Saved labels as \(name)_labels.npy")
  }
}

func getLabelForLetter(letter: String) -> Int {
  
  let features: [String] = "abcdefghiklmnopqrstuvwxy".compactMap({ $0.description })
  
  return features.index(of: letter) ?? -1
}
