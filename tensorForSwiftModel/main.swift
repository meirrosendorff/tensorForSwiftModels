//
//  main.swift
//  tensorForSwiftModel
//
//  Created by Meir Rosendorff on 2019/07/15.
//  Copyright © 2019 Meir Rosendorff. All rights reserved.
//

import Foundation
import Python

let rootPath = "/Users/MRosendorff/DVT_Grad_Program/tensorForSwiftModel/tensorForSwiftModel/"

let dataSet = parseAudioFilesToNumpyArray(dir: rootPath, indexFile: "audioLog.txt", savedFileName: "numpyArrays/test")

let np = Python.import("numpy")

let labelsSaved = np.load(rootPath + "numpyArrays/test_labels.npy")

print(dataSet.labels)
print(labelsSaved)

