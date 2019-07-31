# tensorForSwift ASL Model


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

- Download the dataset from https://www.kaggle.com/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out
- unzip, the dataset has 5 people labeled A-E, either copy all files into one larger directory or train on only one persons hand. Delete all depth files and only keep the .png files.
- Create a folder in your project called fingerSpelling and copy the the directories that you want to train with into there (make sure these directories are labeled a - y (leaving out j and z as the dataset doesnt include these) also create a folder in fingerspelling called nothing and put a whole lot of random images in there. 
- Go into main.swift and edit the root path to be the path of your project.
- Folder Structure is:
  ```
  RootPath
  └── tensorForSwiftModel
    ├── tensorForSwiftModel
    │   ├── coreMlModels
    │   │   └── trainingLogs
    │   ├── fingerSpelling
    │   │   ├── a
    │   │   ├── b
    │   │   ├── c
    │   │   ├── d
    │   │   ├── e
    │   │   ├── f
    │   │   ├── g
    │   │   ├── h
    │   │   ├── i
    │   │   ├── k
    │   │   ├── l
    │   │   ├── m
    │   │   ├── n
    │   │   ├── nothing
    │   │   ├── o
    │   │   ├── p
    │   │   ├── q
    │   │   ├── r
    │   │   ├── s
    │   │   ├── t
    │   │   ├── u
    │   │   ├── v
    │   │   ├── w
    │   │   ├── x
    │   │   └── y
    │   └── numpyArrays
    ├── tensorForSwiftModel.xcodeproj
    │   ├── project.xcworkspace
    │   └── xcuserdata
    └── tensorForSwiftModel.xcworkspace
  ```
- You can now run main.swift and it will build a dataset, save it and then train a model on it.
- You can edit the following parameters to influence your training:
  ```
  dataSetName - The name of the dataset to be created or loaded
  testPercentage - percentage of data to be used for validation
  learningRate - learning rate for a Stochastic Gradient Decent optomizer.
  epochCount - Number of epochs to run for
  numBatches - Number of batches to split the data into (it will be split into numBatches equally sized batches)
  modelName - model name for saving the model.
  ```

## Notes
- After the first run the dataset is saved in numpyArrays/ under the dataset name at the top if the file so if you aren't changing the dataset (like changng the number of files or something like that) then you can comment out the parseImagesToNumpyArray function after the first run as as the Dataset object only needs the dataset name and will then read in these files.
- You can optionally add the parameter maxFiles to parseAudioFilesToNumpyArray to limit the number of files used, this can speed up testing as you arent using the full dataset.
- Your model will be saved at the end of the run in the coreMLModels folder both in .mlmodel and .h5 format, it will also save the model every 50 epochs in coreMLModels/trainingLogs/.
- If you later want to continue traing on an old model you can just give the path to the .h5 file to the ASLModel innitializer using the optional parameter savedModel: String.

## Dependancies
### Python Libraries 
```
os - Reading files
keras - For saving and reading in the model in .h5 format.
coremltools - For saving the model in .mlmodel format.
numpy - numpy stuff
scikit learn - seperating testing an training datasets
matplotlib - graphing output
opencv-python - For reading in and formatting images
random - used for randomly shuffling the files order so that if you don't use all files you still get a good distribution.
```
