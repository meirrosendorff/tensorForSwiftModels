//
//  main.swift
//  tensorForSwiftModel
//
//  Created by Meir Rosendorff on 2019/07/15.
//  Copyright © 2019 Meir Rosendorff. All rights reserved.
//

import Foundation

let rootPath = "/Users/MRosendorff/DVT_Grad_Program/tensorForSwiftModel/tensorForSwiftModel/"
let outputDir = "splitAudio/"

print(AudioFeatureExtractor(fileName: "\(rootPath)\(outputDir)0.wav", numMFCC: 40))

