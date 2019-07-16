//
//  audioFeatureExtractor.swift
//  tensorForSwiftModel
//
//  Created by Meir Rosendorff on 2019/07/16.
//  Copyright © 2019 Meir Rosendorff. All rights reserved.
//

import Foundation
import TensorFlow
import Python

func audioFeatureExtractor(fileName name: String, numMFCC: Int) -> AudioFeature {
  
  let librosa = Python.import("librosa")
  let np  = Python.import("numpy")
  
  let (x, sample_rate) = librosa.load(name).tuple2
  
  let shortTimeFourierTransform = np.abs(librosa.stft(x))
  
  let melFrequencyCepstralCoefficients = np.mean(librosa.feature.mfcc(y: x,
                                                                      sr: sample_rate,
                                                                      n_mfcc: numMFCC).T,
                                                 axis:0)
  let chromagram = np.mean(librosa.feature.chroma_stft(S: shortTimeFourierTransform,
                                               sr: sample_rate).T,
                   axis: 0)
  
  let melScaledPowerSpectogram = np.mean(librosa.feature.melspectrogram(x, sr: sample_rate).T, axis:0)
  
  let spectralContrast = np.mean(librosa.feature.spectral_contrast(S: shortTimeFourierTransform,
                                                                   sr: sample_rate).T,
                                 axis:0)
  
  let tonalCentroidFeatures = np.mean(librosa.feature.tonnetz(y: librosa.effects.harmonic(x),
                                                              
                                                              sr: sample_rate).T,
                                      axis: 0)
  
  return AudioFeature(mfccs: melFrequencyCepstralCoefficients,
                      chroma: chromagram,
                      mel: melScaledPowerSpectogram,
                      contrast: spectralContrast,
                      tonnetz: tonalCentroidFeatures)
}

struct AudioFeature {
  
  let mfccs,chroma,mel,contrast,tonnetz : PythonObject
  
}
