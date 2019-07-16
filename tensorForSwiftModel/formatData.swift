////
////  formatData.swift
////  tensorForSwiftModel
////
////  Created by Meir Rosendorff on 2019/07/15.
////  Copyright © 2019 Meir Rosendorff. All rights reserved.
////
//import Foundation
//import TensorFlow
//import Python
//
//import Foundation
//
//func formatData() {
//
//  let rootPath = "/Users/MRosendorff/DVT_Grad_Program/tensorForSwiftModel/tensorForSwiftModel"
//  let fileDir = "\(rootPath)/respiratory-sound-database/Respiratory_Sound_Database/audio_and_txt_files"
//
//  let os = Python.import("os")
//
//  let fileNames = os.listdir("\(fileDir)")
//
//  let sortedFileNames = fileNames.sorted()
//
//  var audioFiles = [AudioFile]()
//
//  for i in 0..<sortedFileNames.count/2 {
//    guard let infoFile = String(sortedFileNames[2 * i]) else { continue }
//    guard let audioFile = String(sortedFileNames[2 * i + 1]) else { continue }
//    
//    audioFiles.append(AudioFile(info: infoFile, audio: audioFile))
//  }
//
//
//  print(audioFiles)
//
//}
//
//struct AudioFile {
//  let info: String
//  let audio: String
//}
