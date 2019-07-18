import os
import pydub


rootPath = "/Users/MRosendorff/DVT_Grad_Program/tensorForSwiftModel/tensorForSwiftModel/"

fileDir = rootPath + "respiratory-sound-database/Respiratory_Sound_Database/audio_and_txt_files/"

fileNames = os.listdir(fileDir)

fileNames.sort()

fileSets = [(fileNames[2*i], fileNames[2*i+1]) for i in range(int(len(fileNames)/2))]

outputDir = "splitAudio/"

logFile = open(rootPath + "audioLog.txt", "w+")

fileNum = 0

for (info, audio) in fileSets:

    print("Parsing file: " + info)

    info = open(fileDir + info, "r")

    audioFile = pydub.AudioSegment.from_wav(fileDir + audio)

    for line in info:
        values = line.split("\t")
        start = int(float(values[0]) * 1000)
        end = int(float(values[1]) * 1000)
        newAudio = audioFile[start: end]
        newAudio.export(rootPath + outputDir + str(fileNum) + ".wav", format="wav")
        logFile.write(outputDir + str(fileNum) + ".wav" + "\t" + values[2] + "\t" + values[3])

        fileNum += 1

logFile.close()

