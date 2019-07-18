//
//  ParseAudioFiles.swift
//  tensorForSwiftModel
//
//  Created by Meir Rosendorff on 2019/07/16.
//  Copyright © 2019 Meir Rosendorff. All rights reserved.
//

import Foundation
import TensorFlow
import Python

//use max files if you dont want to read in all files
func parseAudioFilesToNumpyArray(dir: String, indexFile fileName: String, savedFileName: String? = nil, maxFiles: Int? = nil) -> (features: PythonObject, labels: PythonObject) {
  
  let io = Python.import("io")
  let np = Python.import("numpy")
  
  let featuresSize = 153
  let numMFCC = 40
  let totalFeatures = featuresSize + numMFCC
  
  let indexFile = io.open(dir + fileName, "r")
  
  var features = np.empty([0,totalFeatures])
  var labels = np.empty(0)

  var numProccesed = 1
  var healthynum = 0
  for line in indexFile {
    
    print("Proccesing audio \(numProccesed)")
    
    guard let lineValues = String(line)?.replacingOccurrences(of: "\n", with: "").split(separator: "\t") else { continue }
    let audioFile = String(lineValues[0])
    
    let label = getLabel(lineValues)
  
    //Uncomment to limit the number of healthy sounds
    
//    if label == 0 {
//      healthynum += 1
//      if healthynum % 3 != 0 {
//        continue
//      }
//    }
    
    let feature = audioFeatureExtractor(fileName: dir + audioFile, numMFCC: numMFCC)
    
    let featureStack = np.hstack([feature.mfccs, feature.chroma, feature.mel, feature.contrast, feature.tonnetz])
    
    features = np.vstack([features, featureStack])

    labels = np.append(labels, label)

    numProccesed += 1
    
    if let limit = maxFiles, numProccesed > limit { break }
    
  }
  
  if let name = savedFileName {
    np.save(dir + "\(name)_features.npy", features)
    print("Saved Features as \(name)_features.npy")
    np.save(dir + "\(name)_labels.npy", labels)
    print("Saved labels as \(name)_labels.npy")
  }
  
  return (features: features, labels: labels)
}

//Returns data label based on wheezing and crackles
/*labesl:
  - 0: Healthy
  - 1: Crackles
  - 2: Wheezing
  - 3: Wheezing and crackles
 */
func getLabel(_ values: [Substring]) -> Int {

  let wheezing = Int(values[1]) ?? 0
  let crackles = Int(values[2]) ?? 0
  
  if wheezing == 1 && crackles == 1 {
    return 3
  } else if wheezing == 1 {
    return 2
  } else if crackles == 1 {
    return 1
  }
  return 0
}


