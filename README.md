# tensorForSwiftBreathingModel


## Tensor flow installation

- Download and install the xCode11 realese for Swift for Tensorflow from https://github.com/tensorflow/swift/blob/master/Installation.md
- Install xcode11 beta on your machine.
- Make a MacOS command line tool project
- Open your project in xCode 11 and set the toolchain to use the one you just downloaded.
- In your projects build settings 
  - Add /Library/Developer/Toolchains/swift-tensorflow-RELEASE-0.4.xctoolchain/usr/lib/swift/macosx to your Runpath Search Paths
  - Change the optimization level in the Swift Compiler - Code generation Section to -O
- Add libtensorflow.so and libtensorflow_framework.so to Link Binary With Libraries in Build Phases, these libraries are located in /Library/Developer/Toolchains/swift-tensorflow-RELEASE-0.4.xctoolchain/usr/lib/swift/macosx
- In File > Project Settings change build system to a legacy build system.

- Try import TensortFlow and import Python and make sure you can compile and run (if you cant cry a little and throw a tantram and mayber xcode will have pity)
- Check which version of python your tensorflow is using by typing 
  ```
  import Python
  print(Python.version)
  ```
  Make sure all python libraries you want to used are installed for that version of python (just pip/pip3 install)

## Running The model trainer

- Download the dataset from https://www.kaggle.com/vbookshelf/respiratory-sound-database
- Format the dataset using the python script formatter.py, go into the script and edit the root directory where the dataset was downloaded to and create the folder splitAudio inside that root directory before running it.
- Go into main.swift and edit the root path to be the path where the data is all stored also create a folder called numpyArrays inside that root path.
- You can now run main.swift and it will build a dataset, save it and then train a model on it.

## Notes
- After the first run the dataset is saved in numpyArrays/ under the dataset name at the top if the file so if you arent changing the dataset (like changng the number of files or something like that) then you can comment out the parseAudioFilesToNumpyArray function after the first run as as the Dataset object only need the dataset name and will then read in these files.
- You can optionally add the parameter maxFiles to parseAudioFilesToNumpyArray to limit the number of files used, this can speed up testing as you arent using the full dataset.

## Dependancies
### Python Libraries 
```
os - Reading files
pydub - Editing Audio Files
numpy - numpy stuff
librosa - feature extraction from audio files.
scikit learn - seperating testing an training datasets
matplotlib - graphing output
```
