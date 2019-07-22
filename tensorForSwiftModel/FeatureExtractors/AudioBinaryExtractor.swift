//
//  AudioBinaryExtractor.swift
//  tensorForSwiftModel
//
//  Created by Meir Rosendorff on 2019/07/18.
//  Copyright © 2019 Meir Rosendorff. All rights reserved.
//

import Foundation
import AVFoundation


func audioBinaryExtractor(fileURl: String,  secondsLimit: Double, sampleRate: Int) -> [Float] {
  
  let url = URL(fileURLWithPath: fileURl)
  let generalSampleRate: Double = Double(sampleRate)

  let file = try! AVAudioFile(forReading: url)
  guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: file.fileFormat.sampleRate, channels: 1, interleaved: false) else { print("awww failed"); return [] }
  
  guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(Int32(generalSampleRate * secondsLimit)) ) else { print("awww failed2"); return []  }
  try! file.read(into: buf)
  
  if Double(buf.frameLength) / buf.format.sampleRate >= secondsLimit { return [] }
  if file.fileFormat.sampleRate != generalSampleRate { return [] }

  let floatArray = Array(UnsafeBufferPointer(start: buf.floatChannelData?[0], count:Int(buf.frameLength)))

  return floatArray + [Float](repeating: 0, count: Int(buf.frameCapacity - buf.frameLength))
}
