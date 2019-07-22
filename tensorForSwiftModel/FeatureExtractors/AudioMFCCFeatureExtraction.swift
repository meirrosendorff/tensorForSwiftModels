//
//  AudioMFCCFeatureExtraction.swift
//  tensorForSwiftModel
//
//  Created by Meir Rosendorff on 2019/07/22.
//  Copyright © 2019 Meir Rosendorff. All rights reserved.
//

import Foundation
import AVFoundation

func audioMFCCFeatureExtractor(fileName fileURl: String, numCoeffs: Int) -> [Double] {
  
  let url = URL(fileURLWithPath: fileURl)
  let numberOfFrames = 262144
  let sampleRate: Float = 44100.0
  
  let specLen = numberOfFrames / 2+1
  let numFilters = numCoeffs + 1
  
  let file = try! AVAudioFile(forReading: url)
  guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: file.fileFormat.sampleRate, channels: 1, interleaved: false) else { print("awww failed"); return [] }
  
  if file.fileFormat.sampleRate != Double(sampleRate) { return [] }
  
  guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(numberOfFrames) ) else { print("awww failed2"); return []  }
  try! file.read(into: buf)
  
  let samples = Array(UnsafeBufferPointer(start: buf.floatChannelData?[0], count:Int(buf.frameLength))) + [Float](repeating: 0, count: Int(buf.frameCapacity - buf.frameLength))

  let fft = FastTransform(withSize: numberOfFrames, sampleRate: sampleRate)
  fft.windowType = WindowType.hanning
  fft.fftForward(samples)

  // Interpoloate the FFT data so there's one band per pixel.
  fft.calculateLinearBands(minFrequency: 0, maxFrequency: fft.nyquistFrequency, numberOfBands: specLen)
  
  let fftBandMagnitudes = fft.bandMagnitudes.map({Double($0)})

  let mfccs = MFCCepstrum(samplingRate: Int(sampleRate), numFilters: numFilters, binSize: specLen, numCoeffs: numCoeffs)
  var mfccData = [Double](repeating: 0.0, count: numCoeffs)
  mfccData = mfccs.getCoefficients(spectralData: fftBandMagnitudes, mfccs: mfccData)
  
  return mfccData
}
